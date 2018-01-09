import pickle

# Loss file
large_gen_loss_file = 'gen_loss_large.pickle'
large_dis_loss_file = 'dis_loss_large.pickle'

with open(large_gen_loss_file, 'rb') as f:
    _list = pickle.load(f)
print(_list)