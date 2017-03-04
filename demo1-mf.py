import mxnet as mx
from movielens_data import get_data_iter, max_id
from matrix_fact import train

ctx = [mx.cpu(0)]

train_test_data = get_data_iter(batch_size=50)
max_user, max_item = max_id('./ml-100k/u.data')

def plain_net(k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user feature lookup
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k) 
    # item feature lookup
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    # predict by the inner product, which is elementwise product and then sum
    pred = user * item
    pred = mx.symbol.sum(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

net1 = plain_net(64)
plot = mx.viz.plot_network(net1)
plot.render('zhfh')

print 'end'