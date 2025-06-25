import os
import cv2
import numpy as np
import sys

labeldict = {chr(ord('A') + i): i for i in range(26)} # temp for handsign dataset
# create a standart for keeping these kinds of data OUTSIDE of the source code
learning_rate=0.01
num_kernels=8
num_layers=2
kernel_size=3
pool_size=2
image_size=28
epochnum=10

input_folder = "dataset"
label_folder = "label"

kernels = [
    [[np.random.randn(kernel_size, kernel_size) * np.sqrt(2 / (kernel_size * kernel_size)) 
      for _ in range(num_kernels)] for _ in range(num_kernels if l > 0 else 1)]
    for l in range(num_layers)
]

def calc_output_size(image_size, num_layers, kernel_size, pool_size):
    size = image_size
    for _ in range(num_layers):
        size = (size - kernel_size + 1) // pool_size
    return size

finmap = calc_output_size(image_size, num_layers, kernel_size, pool_size)
densein = num_kernels * (finmap*finmap)
weights = np.random.randn(len(labeldict), densein) * np.sqrt(2 / densein)
bias = np.zeros(len(labeldict))

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

def forward_pass(img): # TODO: batchnorm for layers is NEEDED
    inputs = [[img]]
    for l in range(num_layers):
        prev_maps = inputs[-1]
        new_maps = []
        for k in range(num_kernels):
            acc = None
            for m_idx, prev in enumerate(prev_maps):
                conv = convolve(prev, kernels[l][m_idx][k])
                if acc is None:
                    acc=conv 
                else: 
                    acc += conv
            act = relu(acc)
            pooled = maxpool(act, pool_size)
            new_maps.append(pooled)
        inputs.append(new_maps)
    return inputs

def backward_pass(inputs, doutput, original_img): #again, batchnorm
    next_grads = doutput
    for l in reversed(range(num_layers)):
        new_grads = [np.zeros_like(inputs[l][i]) for i in range(len(inputs[l]))]
        for k in range(num_kernels):
            out_fmap = inputs[l+1][k]
            act_fmap = relu(out_fmap)
            dpool = maxpool_backward(next_grads[k], act_fmap, pool_size)
            dact = derivrelu(act_fmap) * dpool
            for m_idx in range(len(inputs[l])):
                dinput = kernel_learn(kernels[l][m_idx][k], inputs[l][m_idx], dact)
                new_grads[m_idx] += dinput
        next_grads = new_grads

for epoch in range(epochnum): # huge mess, clean it up
    errors = 0
    files = os.listdir(input_folder)
    for idx, file in enumerate(files):
        img = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        label_file = file.split(".")[0] + ".txt"
        with open(os.path.join(label_folder, label_file)) as f:
            label = labeldict[f.read().strip()]
        input_layers = forward_pass(img)
        final_maps = np.stack(input_layers[-1])
        flat = final_maps.flatten()
        logits = np.dot(weights, flat) + bias
        probs = softmax(logits)
        loss = losscalc(probs, label)
        probs[label] -= 1
        dflat = np.dot(weights.T, probs)
        weights -= learning_rate * np.outer(probs, flat)
        bias -= learning_rate * probs
        dfmap = dflat.reshape(final_maps.shape)
        backward_pass(input_layers, dfmap, img)
        if np.argmax(logits) != label:
            errors += 1
        if idx:
            sys.stdout.write("\033[s")
            print(f"guess: {np.argmax(logits) == label} | acc: %{100 - (errors/(idx+1))*100} | loss: {loss}")
            print(f"progress: {idx+1}/{len(files)} | %{(idx+1)/len(files)}")
            sys.stdout.write("\033[u")
    print("\n\n")
    print(f"epoch {epoch+1} accuracy: %{100 - (errors / len(files)) * 100}\n")

np.savetxt('weights', weights, delimiter=',')
np.savetxt('bias', bias, delimiter=',')
with open("kernels", "w") as f:
    for x in kernels:
        for y in x:
            for kern in y:
                f.write(str(kern.tolist()) + "\n") # keep a standart for keeping?