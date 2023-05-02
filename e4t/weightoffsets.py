import torch
from torch import nn
from einops import rearrange, repeat

# class WeightOffsets(nn.Module):
#     def __init__(self, row_dim, column_dim):
#         super().__init__()
#         self.v = nn.Parameter(torch.ones(1))  # v0 dim == 1
#         self.linear1 = nn.Linear(1, row_dim)
#         self.linear2 = nn.Linear(1, column_dim)
#         self.linear_column = nn.Linear(row_dim, row_dim)
#         self.linear_row = nn.Linear(column_dim, column_dim)

#     def forward(self, input=None):
#         vx = self.linear1(self.v) # (row_dim)
#         vy = self.linear2(self.v) # (column_dim)
#         # matrix multiplication -> (row_dim, column_dim)
#         v_matrix = vx.unsqueeze(0).T * vy.unsqueeze(0)
#         # columnwise
#         v_matrix = self.linear_column(v_matrix.T)
#         # rowwise
#         v_matrix = self.linear_row(v_matrix.T)
#         return v_matrix.T

""" weight offset configs: dW = a b.T
    - base1: lin1 (v -> b), lin2 (v -> a), lin3, lin4
    - base2: lin1 (v -> b), lin2 (v -> a); w <- w + wo
    - more1: lin1 (x -> b), lin2 (y -> a), lin3, lin4
    - more2: lin1 (v -> b), lin2 (y -> a), lin3, lin4
    - more3: lin1 (x + v -> b), lin2 (y + v -> a), lin3, lin4
    - more4: lin1 (v -> b), lin2 (y -> a); w <- w + wo
    - more5: lin1 (y -> b), lin2 (y -> a), lin3, lin4 [hyper]
    - more6: lin1 (y -> b), lin2 (y -> a); w <- w + wo
"""

class WeightOffsets(nn.Module):
    def __init__(self, row_dim, column_dim, wo_config='base1', wo_vdim=512,
                 wo_embed_dim=768, wo_zero_init=True, wo_detach_input=True,
                 **kwargs):
        super().__init__()
        self.wo_config = wo_config
        self.wo_detach_input = wo_detach_input
        self.row_dim = row_dim
        self.column_dim = column_dim
        if wo_config == 'base1':
            self.v = nn.Parameter(torch.randn(wo_vdim)*1.)
            self.linear1 = nn.Linear(wo_vdim, row_dim)
            self.linear2 = nn.Linear(wo_vdim, column_dim)
            self.linear_column = nn.Linear(row_dim, row_dim)
            self.linear_row = nn.Linear(column_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear_row.weight.data.zero_()
                    self.linear_row.bias.data.zero_()
        elif wo_config == 'base2':
            self.v = nn.Parameter(torch.randn(wo_vdim)*1.)
            self.linear1 = nn.Linear(wo_vdim, row_dim)
            self.linear2 = nn.Linear(wo_vdim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    # LoRA init B as zero (BAx)
                    self.linear2.weight.data.zero_()
                    self.linear2.bias.data.zero_()
        elif wo_config == 'more1':
            self.linear1 = nn.Linear(row_dim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim, column_dim)
            self.linear_column = nn.Linear(row_dim, row_dim)
            self.linear_row = nn.Linear(column_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear_row.weight.data.zero_()
                    self.linear_row.bias.data.zero_()
        elif wo_config == 'more2':
            self.v = nn.Parameter(torch.randn(wo_vdim)*1.)
            self.linear1 = nn.Linear(wo_vdim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim, column_dim)
            self.linear_column = nn.Linear(row_dim, row_dim)
            self.linear_row = nn.Linear(column_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear_row.weight.data.zero_()
                    self.linear_row.bias.data.zero_()
        elif wo_config == 'more3':
            self.v = nn.Parameter(torch.randn(wo_vdim)*1.)
            self.linear1 = nn.Linear(row_dim+wo_vdim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim+wo_vdim, column_dim)
            self.linear_column = nn.Linear(row_dim, row_dim)
            self.linear_row = nn.Linear(column_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear_row.weight.data.zero_()
                    self.linear_row.bias.data.zero_()
        elif wo_config == 'more4':
            self.v = nn.Parameter(torch.randn(wo_vdim)*1.)
            self.linear1 = nn.Linear(wo_vdim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear2.weight.data.zero_()
                    self.linear2.bias.data.zero_()
        elif wo_config == 'more5':
            self.linear1 = nn.Linear(wo_embed_dim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim, column_dim)
            self.linear_column = nn.Linear(row_dim, row_dim)
            self.linear_row = nn.Linear(column_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear_row.weight.data.zero_()
                    self.linear_row.bias.data.zero_()
        elif wo_config == 'more6':
            self.linear1 = nn.Linear(wo_embed_dim, row_dim)
            self.linear2 = nn.Linear(wo_embed_dim, column_dim)
            if wo_zero_init:
                with torch.no_grad():
                    self.linear2.weight.data.zero_()
                    self.linear2.bias.data.zero_()
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor = None, embed=None, bypass=False):
        if bypass:
            return 0
        batch_size = input.shape[0]
        if self.wo_config == 'base1':
            vx = self.linear1(self.v) # (row_dim)
            vy = self.linear2(self.v) # (column_dim)
            # matrix multiplication -> (row_dim, column_dim)
            v_matrix = vx.unsqueeze(0).T * vy.unsqueeze(0)
            # columnwise
            v_matrix = self.linear_column(v_matrix.T)
            # rowwise
            v_matrix = self.linear_row(v_matrix.T)
            return v_matrix.T  # (column_dim, row_dim)
        elif self.wo_config == 'base2':
            vx = self.linear1(self.v)
            vy = self.linear2(self.v)
            v_matrix = vx.unsqueeze(0).T * vy.unsqueeze(0)
            return v_matrix.T
        elif self.wo_config == 'more1':
            input = input.mean(dim=1)
            if self.wo_detach_input:
                input = input.detach()
            vx = self.linear1(input) # (row_dim)
            vy = self.linear2(embed) # (column_dim)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            v_matrix = self.linear_column(rearrange(v_matrix, 'b i j -> b j i'))
            v_matrix = self.linear_row(rearrange(v_matrix, 'b i j -> b j i'))
            return rearrange(v_matrix, 'b i j -> b j i')
        elif self.wo_config == 'more2':
            vx = self.linear1(self.v).unsqueeze(0)
            vy = self.linear2(embed)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            v_matrix = self.linear_column(rearrange(v_matrix, 'b i j -> b j i'))
            v_matrix = self.linear_row(rearrange(v_matrix, 'b i j -> b j i'))
            return rearrange(v_matrix, 'b i j -> b j i')
        elif self.wo_config == 'more3':
            input = input.mean(dim=1)
            if self.wo_detach_input:
                input = input.detach()
            input_x = torch.cat([input, self.v.expand(input.shape[0], -1)], dim=-1)
            input_y = torch.cat([embed, self.v.expand(embed.shape[0], -1)], dim=-1)
            vx = self.linear1(input_x)
            vy = self.linear2(input_y)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            v_matrix = self.linear_column(rearrange(v_matrix, 'b i j -> b j i'))
            v_matrix = self.linear_row(rearrange(v_matrix, 'b i j -> b j i'))
            return rearrange(v_matrix, 'b i j -> b j i')
        elif self.wo_config == 'more4':
            vx = self.linear1(self.v).unsqueeze(0)
            vy = self.linear2(embed)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            return rearrange(v_matrix, 'b i j -> b j i')
        elif self.wo_config == 'more5':
            vx = self.linear1(embed)
            vy = self.linear2(embed)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            v_matrix = self.linear_column(rearrange(v_matrix, 'b i j -> b j i'))
            v_matrix = self.linear_row(rearrange(v_matrix, 'b i j -> b j i'))
            return rearrange(v_matrix, 'b i j -> b j i')
        elif self.wo_config == 'more6':
            vx = self.linear1(embed)
            vy = self.linear2(embed)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            return rearrange(v_matrix, 'b i j -> b j i')
        else:
            return 0


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wo = WeightOffsets(32, 16)
        self.linear = nn.Linear(32, 16)
        self.init_weight = None
        self.wo_out = None
        self.linear.weight.register_hook(self.wo_backward)

    def wo_backward(self, grad):
        print("grad:", grad)
        grad = grad * self.init_weight
        self.wo_out.backward(grad)

    def update_weight(self):
        if self.init_weight is None:
            self.init_weight = self.linear.weight.data.clone()
        self.wo_out = self.wo()
        self.linear.weight.data = self.init_weight * (1 + self.wo_out)

    def forward(self, x):
        self.update_weight()
        y = self.linear(x)
        return y


if __name__ == '__main__':
    model = Model()
    # model = WeightOffsets(32, 16)
    # linear = torch.nn.Linear(32, 16)
    # # linear.requires_grad_(False)
    # init_weight = linear.weight.data.clone()
    optimizer = torch.optim.AdamW(model.wo.parameters(), lr=0.01)
    # train!
    model.train()
    optimizer.zero_grad()

    x = torch.randn(2, 32)
    y = torch.randn(2, 16)
    # wo_weight = model()
    print(model.wo.v)
    # linear.weight.data = init_weight * (1 + wo_weight)
    # out = linear(x)
    out = model(x)
    loss = nn.functional.mse_loss(y, out)
    # loss = wo_weight.sum()
    print("loss:", loss)
    loss.backward()
    # grad = linear.weight.grad * init_weight
    # wo_weight.backward(grad)
    optimizer.step()
    print(model.wo.v)