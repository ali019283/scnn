import os
import cv2
import numpy as np
import sys
import pickle

labeldict = {chr(ord('A') + i): i for i in range(26)} # temp for handsign dataset
# create a standart for keeping these kinds of data OUTSIDE of the source code
learning_rate=0.003
num_kernels=4
num_layers=2
# num layer 2 
# num kernel 2
# or the value for exponantial growth with how many times it is exp grown
# exp: +4 layer +6 kernel every 4 pools 2 times
#init_kernel =  []
kernel_size=3
pool_size=2
image_size=28
epochnum=3
val_folder = "dataset"
input_folder = "dataset"
label_folder = "label"
read_mode=0
kernels = []
in_channels = 1
for l in range(num_layers):
    out_channels = num_kernels * (num_kernels ** l)
    kernels.append([
        [np.random.randn(kernel_size, kernel_size) * np.sqrt(2 / (kernel_size * kernel_size))
         for _ in range(out_channels)]
        for _ in range(in_channels)
    ])
    in_channels = out_channels

"""gam = [ 
    [[np.full((kernel_size, kernel_size), 1) * np.sqrt(2 / (kernel_size * kernel_size)) 
       for _ in range(num_kernels)] for _ in range(num_kernels if l > 0 else 1)]
    for l in range (num_layers)
]
beta = [
    [[np.full((kernel_size, kernel_size), 0) * np.sqrt(2 / (kernel_size * kernel_size)) 
        for _ in range(num_kernels)] for _ in range(num_kernels if l > 0 else 1)]
    for l in range (num_layers)
]
"""
def calc_output_size(image_size, num_layers, kernel_size, pool_size):
    size = image_size
    for _ in range(num_layers):
        size = (size - kernel_size + 1) // pool_size
    return size


def relu(x): return np.maximum(0, x)

def derivrelu(x): return (x > 0).astype(float)

def convolve(img, kernel): # def must be written in C
    h, w = img.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

def maxpool(img, size):
    h, w = img.shape
    out_h = h // size
    out_w = w // size
    pooled = np.zeros((out_h, out_w))
    for i in range(0, out_h * size, size):
        for j in range(0, out_w * size, size):
            pooled[i // size, j // size] = np.max(img[i:i+size, j:j+size])
    return pooled

def maxpool_backward(dout, img, size): # not even the real backward? keep track of maxpool maybe?
    h, w = img.shape
    dimg = np.zeros_like(img)
    for i in range(0, h, size):
        for j in range(0, w, size):
            window = img[i:i+size, j:j+size]
            max_val = np.max(window)
            for m in range(window.shape[0]):
                for n in range(window.shape[1]):
                    if window[m, n] == max_val:
                        dimg[i+m, j+n] = dout[i//size, j//size]
                        break
    return dimg

"""def batch_norm_dense(x, gam, beta, eps = 1e-5):
    mean = nd.mean(x, axis = 0)
    var = nd.mean((x-mean)**2, axis = 0)
    out = (gam * ((x-mean) * (1.0 / nd.sqrt(var + eps, axis=0)))) + beta
    cache = [gam, beta, var, np.zeros(x.shape)]
    return out, cache

def batch_norm_backw(x, gam, beta, eps = 1e-5):
    ADD IT LATEERR        

"""
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def losscalc(probs, label): 
    return -np.log(probs[label] + 1e-9)

def kernel_learn(kernel, inp, dfeature): # there must be a more stabile way
    kh, kw = kernel.shape
    grad = np.zeros_like(kernel)
    dinput = np.zeros_like(inp)
    for i in range(dfeature.shape[0]):
        for j in range(dfeature.shape[1]):
            patch = inp[i:i+kh, j:j+kw]
            grad += patch * dfeature[i, j]
            dinput[i:i+kh, j:j+kw] += kernel * dfeature[i, j]
    kernel -= learning_rate * grad
    return dinput

def forward_pass(img):
    inputs = [[img]]
    for l in range(num_layers):
        prev_maps = inputs[-1]
        in_channels = len(prev_maps)
        out_channels = len(kernels[l][0])
        new_maps = []
        for out_k in range(out_channels):
            acc = None
            for in_k in range(in_channels):
                conv = convolve(prev_maps[in_k], kernels[l][in_k][out_k])
                if acc is None:
                    acc = conv
                else:
                    acc += conv
            act = relu(acc)
            pooled = maxpool(act, pool_size)
            new_maps.append(pooled)
        inputs.append(new_maps)
    return inputs

def backward_pass(inputs, doutput, original_img):
    next_grads = doutput
    for l in reversed(range(num_layers)):
        in_channels = len(inputs[l])
        out_channels = len(inputs[l+1])
        new_grads = [np.zeros_like(inputs[l][i]) for i in range(in_channels)]
        for out_k in range(out_channels):
            out_fmap = inputs[l+1][out_k]
            act_fmap = relu(out_fmap)
            dpool = maxpool_backward(next_grads[out_k], act_fmap, pool_size)
            dact = derivrelu(act_fmap) * dpool
            for in_k in range(in_channels):
                dinput = kernel_learn(kernels[l][in_k][out_k], inputs[l][in_k], dact)
                new_grads[in_k] += dinput
        next_grads = new_grads

cache_batch = [] 
dummy = np.zeros((image_size, image_size))
output = forward_pass(dummy)
final_maps = np.stack(output[-1])
densein = final_maps.size
weights = np.random.randn(len(labeldict), densein) * np.sqrt(2 / densein) * learning_rate # im tired, this is a bad way ik but its the only one i got rn, FIX LATER

if(read_mode):
    with open("weights.pickle", "rb") as f:
        weights = pickle.load(f)
    with open("kernels.pickle", "rb") as f:
        kernels = pickle.load(f)    
    files = os.listdir(val_folder)
    for idx, file in enumerate(files):
        img = cv2.imread(os.path.join(val_folder, file), cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        label_file = file.split(".")[0] + ".txt"
        with open(os.path.join(label_folder, label_file)) as f:
            label=labeldict[f.read().strip()]
        input_layers = forward_pass(img)
        final_maps = np.stack(input_layers[-1])
        flat = final_maps.flatten()
        logits = np.dot(weights, flat)
        probs = softmax(logits)
        print(f"{probs} : {np.argmax(probs)} : {label} : {file}\n")

for epoch in range(epochnum): # huge mess, clean it up
    if (epoch == 0):
        prev_epoch_acc = 0
    errors = []
    files = os.listdir(input_folder)
    for idx, file in enumerate(files):
        #init simplifying kernel here
        img = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        label_file = file.split(".")[0] + ".txt"
        with open(os.path.join(label_folder, label_file)) as f:
            label = labeldict[f.read().strip()]
        input_layers = forward_pass(img)
        final_maps = np.stack(input_layers[-1])
        flat = final_maps.flatten()
        logits = np.dot(weights, flat)
        probs = softmax(logits)
        loss = losscalc(probs, label)
        probs[label] -= 1
        dflat = np.dot(weights.T, probs)
        weights -= learning_rate * np.outer(probs, flat)
        dfmap = dflat.reshape(final_maps.shape)
        backward_pass(input_layers, dfmap, img)
        if np.argmax(logits) != label:
            errors.append(1)
        else:
            errors.append(0)
        if idx and errors:
            sys.stdout.write("\033[s")
            print(f"guess: {np.argmax(logits) == label} | 100 acc: %{100-np.sum(errors[-100:])} | total acc: %{100 - (np.sum(errors)/(idx+1)*100)} | loss: {loss}")
            print(f"progress: {idx}/{len(files)} | %{(idx)/len(files)*100}")
            sys.stdout.write("\033[u")
        if(100-np.sum(errors[-100:])<(100-(np.sum(errors)/(idx+1)*100)) and idx % 100):
            #learning_rate*=0.99999 if you want continous learning rate adjustment but really not needed that much
            0
        else:
            0
            #learning_rate*=1.001
        if idx == 100:
            with open('kernels1.pickle', 'wb') as handle:
                pickle.dump(kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if idx == 200:
            with open('kernel2.pickle', 'wb') as handle:
                pickle.dump(kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if idx > len(files)-10:
            print(f"{np.argmax(logits)}  {label}")
    print("\n\n")
    print(f"epoch {epoch+1} accuracy: %{100 - (np.sum(errors) / len(files)) * 100}\n")
    if(prev_epoch_acc > np.sum(errors)/len(files)):
        learning_rate *=0.80
    else:
        learning_rate *=1.20
    
    prev_epoch_acc = (np.sum(errors)/len(files))

with open('kernels.pickle', 'wb') as handle:
    pickle.dump(kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('weights.pickle', 'wb') as handle:
    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

