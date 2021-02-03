import h5py

def load_stock_weights(model, path):
    n2v = {}
    with h5py.File(path) as f:
        for key in f:
            n2v[key] = f[key].value

    for i, x in enumerate(model.variables):
        name = x.name[:-2]
        name = name.replace('/', '|')
        if name in n2v:
            x.assign(n2v[name])
            # print("%d:%s %s Y" % (i, x.name, str(x.shape)))
        # else:
        #     print("%d:%s %s N" % (i, x.name, str(x.shape)))
    for i, x in enumerate(model.trainable_variables):
        name = x.name[:-2]
        name = name.replace('/', '|')
        if name not in n2v:
            print("%d:%s %s N" % (i, x.name, str(x.shape)))
    return model

def get_value(data, name):
    rt = []
    names = []
    for key in data:
        try:
            names.append(name+'.'+key)
            rt.append(data[key].value)
        except:
            names_, rt_ = get_value(data[key], name+'.'+key)
            names.extend(names_)
            rt.extend(rt_)
    return names, rt

    return 
if __name__=='__main__':
    n2v = {}
    with h5py.File('./checkpoint/yolo4_voc_weights.h5') as f:
        names, rt = get_value(f,'')

    print('')