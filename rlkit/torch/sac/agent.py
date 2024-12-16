import numpy as np

import torch
import copy
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu



def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 club_input_dim,
                 context_encoder,
                 club_model,
                 context_discriminator,
                 # Bisimulation_ratio,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.club_input_dim = club_input_dim

        self.context_encoder = context_encoder
        self.club_model = club_model
        self.context_discriminator = context_discriminator
        self.W = nn.ParameterList([nn.Parameter(torch.rand(self.latent_dim, self.latent_dim), requires_grad = True)]).cuda()
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.register_buffer('params', torch.zeros(1, latent_dim))

        self.register_buffer('club_z_params', torch.zeros(1, latent_dim))
        self.register_buffer('club_z_means', torch.zeros(1, latent_dim))
        self.register_buffer('club_z_vars', torch.zeros(1, latent_dim))

        self.register_buffer('z_params', torch.zeros(1, latent_dim))

        self.register_buffer('z_means_score', torch.zeros(1, latent_dim))
        self.register_buffer('club_z_means_score', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var

        self.context = None
        self.params = None

        self.club_z_params = None
        self.club_z_means = None
        self.club_z_vars = None

        self.z_params = None
        self.z_means_score = None
        self.club_z_means_score = None
        # sample a new z from the prior
        self.sample_z()


    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        # True
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:
            self.is_onlineadapt_max_score = score
            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])

    def fix_update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:

            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])

    def set_z(self, means, vars):
        self.z_means = means
        self.z_vars = vars
        self.sample_z()

    def set_z_sample(self, z):
        self.z = z

    def set_onlineadapt_z_sample(self):
        self.z = self.is_onlineadapt_max_z_sample

    def set_onlineadapt_update_context(self):
        self.context = self.is_onlineadapt_max_upd_context
        self.infer_posterior(self.context)

    def clear_onlineadapt_max(self):
        self.is_onlineadapt_max_score = -1e8
        self.is_onlineadapt_max_context = None
        self.is_onlineadapt_max_upd_context = None
        self.is_onlineadapt_max_z = None
        self.is_onlineadapt_max_z_sample = None
        self.old_onlineadapt_max_score = -1e8

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r=np.array(r)
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
 
    def update_context_dict(self, batch_dict, env):
        ''' append context dictionary containing single/multiple transitions to the current context '''
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            r = batch_dict['rewards']
            sr = []
            for r_entry in r:
                sr.append(env.sparsify_rewards(r_entry))
            r = ptu.from_numpy(np.array(sr)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, next_o], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), 0.05*ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        params = self.context_encoder(context)

        params = params.view(context.size(0), -1, self.context_encoder.output_size)


        club_params = self.club_model(context[:, :, :self.club_input_dim])
        club_params = club_params.view(context.size(0), params.shape[1], -1)


        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
            # self.task_indices = np.zeros((task_num,))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)

        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])

            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])

        else:
            self.params = params
            self.z_means = torch.mean(params, dim=1) # dim: task, batch, feature (latent dim)
            self.z_vars = torch.std(params, dim=1)

            self.club_z_params = club_params[:, :, :self.latent_dim]
            self.club_z_means = torch.mean(self.club_z_params, dim=1)
            self.club_z_vars = club_params[:, :, self.latent_dim:]


        self.sample_z()


    def only_infer_context_embeddings(self, context, b):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)

        club_params = self.club_model(context[:, :, :self.club_input_dim])
        club_params = club_params.view(params.shape[0], params.shape[1], -1)

        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])

            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
                          zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z_embedding = torch.stack(z)

            return z_embeddings
        else:
            z_embeddings = torch.mean(params, dim=1)
            z_embeddings_means = z_embeddings
            z_embeddings = [z.repeat(b, 1) for z in z_embeddings]
            z_embeddings = torch.cat(z_embeddings, dim=0)

            z_params_score = self.context_discriminator(params)
            z_means_score = self.context_discriminator(z_embeddings_means)

            club_params = club_params[:, :, :self.latent_dim]
            club_embeddings = torch.mean(club_params, dim=1)
            club_embeddings_means = club_embeddings
            club_embeddings = [z.repeat(b, 1) for z in club_embeddings]
            club_embeddings = torch.cat(club_embeddings, dim=0)

            club_params_score = self.context_discriminator(club_params)
            club_means_score = self.context_discriminator(club_embeddings_means)

            combine_params, _, _ = self.context_discriminator.compute_softmax_result(
                params, z_params_score, club_params, club_params_score
            )
            combine_z_means, _, _ = self.context_discriminator.compute_softmax_result(
                z_embeddings_means, z_means_score, club_embeddings_means, club_means_score
            )

            return params, z_embeddings, z_embeddings_means, \
                   club_params, club_embeddings, club_embeddings_means, \
                   combine_params, combine_z_means

    def compute_club_model_loss(self, context):
        z_target = self.context_encoder(context).detach()
        z_param = self.club_model(context[:, :, :self.club_input_dim])
        z_mean = z_param[:, :, :self.latent_dim]
        z_var = F.softplus(z_param[:, :, self.latent_dim:])
        club_model_loss = (
                (z_target - z_mean) ** 2 / (2 * z_var) + torch.log(torch.sqrt(z_var))).mean()
        return club_model_loss

    def compute_club_loss(self, context):
        z_target = self.context_encoder(context)
        z_param = self.club_model(context[:, :, :self.club_input_dim]).detach()
        z_mean = z_param[:, :, :self.latent_dim]
        z_var = F.softplus(z_param[:, :, self.latent_dim:])
        z_t, z_b, _ = z_mean.size()
        position = - ((z_target - z_mean) ** 2 / z_var).mean()
        z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b ** 2, -1)
        z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b ** 2, -1)
        z_target_repeat = z_target.repeat(1, z_b, 1)
        negative = - ((z_target_repeat - z_mean_expand) ** 2 / z_var_expand).mean()
        club_loss = (position - negative)

        return club_loss

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            if self.z_means is not None and self.club_z_means is not None:
                z_means_score = self.context_discriminator(self.z_means)
                club_z_means_score = self.context_discriminator(self.club_z_means)
                self.z, self.z_means_score, self.club_z_means_score = self.context_discriminator.compute_softmax_result(
                    self.z_means, z_means_score,
                    self.club_z_means, club_z_means_score
                )

                z_params_score = self.context_discriminator(self.params)
                club_z_params_score = self.context_discriminator(self.club_z_params)
                self.z_params, _, _ = self.context_discriminator.compute_softmax_result(
                    self.params, z_params_score,
                    self.club_z_params, club_z_params_score
                )
            else:
                self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def get_action_with_zero_z(self, obs, task_z):
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        in_ = torch.cat([obs, task_z], dim=1)

        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        return policy_outputs[0]

    def forward(self, obs, context, task_indices=None, for_update=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        in_ = torch.cat([obs, task_z], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        z_means_repeat = [z.repeat(b, 1) for z in self.z_means]
        z_means_repeat = torch.cat(z_means_repeat, dim=0)

        club_means_repeat = [z.repeat(b, 1) for z in self.club_z_means]
        club_means_repeat = torch.cat(club_means_repeat, dim=0)

        if for_update:
            if not self.use_ib:
                task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
                task_z_vars = torch.cat(task_z_vars, dim=0)
                return policy_outputs, task_z, task_z_vars, self.z, \
                       self.params, self.z_means, z_means_repeat, \
                       self.club_z_params, self.club_z_means, club_means_repeat, \
                       self.z_params, self.z_means_score, self.club_z_means_score
            return policy_outputs, task_z, self.z, \
                   self.params, self.z_means, z_means_repeat, \
                   self.club_z_params, self.club_z_means, club_means_repeat, \
                   self.z_params, self.z_means_score, self.club_z_means_score
        else:
            if not self.use_ib:
                task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
                task_z_vars = torch.cat(task_z_vars, dim=0)
                return policy_outputs, task_z, task_z_vars, self.params
            return policy_outputs, task_z, self.params

    def log_diagnostics(self, eval_statistics):
        # adds logging data about encodings to eval_statistics
        # z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        

        for i in range(len(self.z_means[0])):
            z_mean = ptu.get_numpy(self.z_means[0][i])
            name = 'Z mean eval' + str(i)
            eval_statistics[name] = z_mean
        print("z_vars: ", self.z_vars)
        print("z_vars.shape: ", self.z_vars.shape)
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z variance eval'] = z_sig

        # eval_statistics['Z mean eval'] = z_mean
        # eval_statistics['Z variance eval'] = z_sig
        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        # return [self.context_encoder, self.context_score_encoder, self.policy]
        return [self.context_encoder, self.policy, self.club_model, self.context_discriminator]


class OldPEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        #self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_onlineadapt_max(self, score, context):
        if score > self.is_onlineadapt_max_score:
            self.is_onlineadapt_max_score = score
            self.is_onlineadapt_max_context = self.context
            self.is_onlineadapt_max_z = (self.z_means, self.z_vars)
            self.is_onlineadapt_max_z_sample = self.z
            for c in context:
                self.update_context(c)
            self.infer_posterior(self.context)
            self.is_onlineadapt_max_upd_context = self.context
        else:
            self.context = self.is_onlineadapt_max_context
            self.set_z(self.is_onlineadapt_max_z[0], self.is_onlineadapt_max_z[1])

    def set_z(self, means, vars):
        self.z_means = means
        self.z_vars = vars
        self.sample_z()

    def set_z_sample(self, z):
        self.z = z

    def set_onlineadapt_z_sample(self):
        self.z = self.is_onlineadapt_max_z_sample

    def set_onlineadapt_update_context(self):
        self.context = self.is_onlineadapt_max_upd_context
        self.infer_posterior(self.context)

    def clear_onlineadapt_max(self):
        self.is_onlineadapt_max_score = -1e8
        self.is_onlineadapt_max_context = None
        self.is_onlineadapt_max_upd_context = None
        self.is_onlineadapt_max_z = None
        self.is_onlineadapt_max_z_sample = None
        self.old_onlineadapt_max_score =  -1e8

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r=np.array(r)
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            # permutation invariant encoding
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, task_indices=None):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z

        # self.meta_batch * self.batch_size * dim(obs)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1) # in focal these does not use detach()
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]
