from helper import *
from model.message_passing import MessagePassing

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.w_val		= get_param((in_channels, out_channels))
		self.w_key		= get_param((in_channels, out_channels))
		self.w_qry		= get_param((in_channels, out_channels))
		self.w_p 		= get_param((1, in_channels))
		self.loop_rel 		= get_param((1, in_channels))
		self.loop_val		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, edge_val, rel_embed, val_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		val_embed = torch.cat([val_embed, self.loop_val], dim=0)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)
		

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)
		self.loop_val_type   = torch.full((num_ent,), val_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.norm     = self.compute_norm(edge_index,  num_ent)
		
		in_res		= self.propagate('add', edge_index, x=x, edge_type=edge_type, edge_val=edge_val, rel_embed=rel_embed, val_embed = val_embed, edge_norm=self.norm, 	mode='in')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, edge_val=self.loop_val_type, rel_embed=rel_embed, val_embed = val_embed, edge_norm=None, 		mode='loop')
		out		= self.drop(in_res)*(1/2) + loop_res*(1/2)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1], torch.matmul(val_embed, self.w_val)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		#self attention on entity and relation
		a = torch.mm(ent_embed, self.w_key) + torch.mm(rel_embed, self.w_qry)
		m1 = nn.Sigmoid()
		a = m1(a)
		a = torch.mm(a, self.w_p.t())
		m2 = nn.Softmax(dim = 1)
		a = m2(a)

		trans_embed = a.t() * (ent_embed + rel_embed)


		return trans_embed

	def val_transform(self, rel_emb, val_emb):
		trans_embed  = rel_emb + val_emb
		
		return trans_embed



	def message(self, x_j, edge_type, edge_val, rel_embed, val_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		val_emb = torch.index_select(val_embed, 0, edge_val)
		rel_val = self.val_transform(rel_emb, val_emb)
		xj_rel  = self.rel_transform(x_j, rel_val)
		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
