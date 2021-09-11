import torch
from torch.nn import Module


class DissimilarityModule(Module):
    def __init__(self, weights_per_tensor=256):
        super(DissimilarityModule, self).__init__()

        self.weights_per_tensor = weights_per_tensor
        # Define weights initialized with ones
        vector_ones = torch.ones((weights_per_tensor))
        vector_zeros = torch.zeros((weights_per_tensor))

        # Pesos usados no passo 1
        self.weights_ref = torch.nn.Parameter(vector_ones.squeeze())
        self.weights_tar = torch.nn.Parameter(vector_ones.squeeze())
        # Bias usados no passo 3
        self.bias_diff = torch.nn.Parameter(
            (0.01 + vector_zeros).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        # Pesos usados no passo 4
        self.weights_channels = torch.nn.Parameter(vector_ones)

    def forward(self, ref_tensor, tar_tensor):
        # Aplicando a DISTANCIA EUCLIDIANA PADRÃO (sem aprender pesos)
        # out = ref_tensor - tar_tensor
        # out = out**2
        # out = out.sum(axis=1)
        # out = torch.sqrt(out)

        # Aplicando a DISTANCIA EUCLIDIANA PADRÃO (aprendendo pesos)
        # ref_tensor = torch.einsum('bchw,c->bchw', [ref_tensor, self.weights_ref])
        # tar_tensor = torch.einsum('bchw,c->bchw', [tar_tensor, self.weights_tar])
        # out = ref_tensor - tar_tensor
        # out = out**2
        # out = out.sum(axis=1)
        # out = torch.sqrt(out)

        # Abaixo o código original, com distância genérica
        # Passo 1: Pondera cada canal dos tensores de referência e alvo com pesos
        ref_tensor = torch.einsum('bchw,c->bchw', [ref_tensor, self.weights_ref])
        tar_tensor = torch.einsum('bchw,c->bchw', [tar_tensor, self.weights_tar])
        # Passo 2: Subtrai tensores
        out = ref_tensor - tar_tensor
        # Passo 3: Soma cada canal com bias
        out = out + self.bias_diff
        # pass by a non-linearity
        out = torch.tanh(out)
        # square the result
        out = out**2
        # Passo 4: Multiplica cada canal por um peso
        out = torch.einsum('bchw,c->bchw', out, self.weights_channels)
        # soma os canais
        out = out.sum(axis=1)
        return out


class N_TensorsDifference(Module):
    def __init__(self, n_pairs, weights_per_tensor=256, gamma_sigmoid=10.):
        super(N_TensorsDifference, self).__init__()
        self.n_pairs = n_pairs
        self.weights_per_tensor = weights_per_tensor
        self.gamma_sigmoid = gamma_sigmoid

        self.branches = [DissimilarityModule(self.weights_per_tensor) for n in range(n_pairs)]
        for i, branch in enumerate(self.branches):
            self.add_module(str(i), branch)

        #########################################
        # Parâmetros usados nos passos 6 ao 9   #
        #########################################
        if self.n_pairs > 1:
            self.combination_weights = torch.nn.Parameter(
                torch.ones((self.n_pairs)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
        # self.combination_bias = torch.nn.Parameter(torch.tensor(0.01))
        self.combination_bias = torch.nn.Parameter(torch.tensor(-2.15))
        # self.combination_bias = torch.nn.Parameter(torch.tensor(-3.15))

    def forward(self, ref_tensor, tar_tensor):
        # para cada par de pesos
        out = torch.stack([branch(ref_tensor, tar_tensor) for branch in self.branches], 0)

        if self.n_pairs > 1:
            # passo 6
            out = out * self.combination_weights
            # passo 7
            out = out.sum(axis=0)
        # Passos 8 e 9
        out = torch.sigmoid(self.gamma_sigmoid * (out + self.combination_bias))
        return out.squeeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            p.grad = None

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
