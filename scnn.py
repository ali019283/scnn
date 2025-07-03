import os
import cv2
import numpy as np
import sys
import pickle
import pyautogui

labeldict = {"W":0, "O":1, "L":2, "Y":3, "A":4, "I":5, "U": 6}# temp for handsign dataset
# create a standart for keeping these kinds of data OUTSIDE of the source code
bboxlearning_rate=0.01
learning_rate=0.0001
num_kernels= 2
num_layers= 2
bboxnum_kernels= 1
bboxnum_layers= 2
# num layer 2 
# num kernel 2
# or the value for exponantial growth with how many times it is exp grown
# exp: +4 layer +6 kernel every 4 pools 2 times
# init_kernel =  []
kernel_size=3
pool_size=2
image_size=28
epochnum=2
val_folder = "dataset"
input_folder = "dataset"
label_folder = "label"
kernels = []
bbox_kernels = []
in_channels = 1
for l in range(num_layers):
    out_channels = num_kernels * (num_kernels ** l)
    kernels.append([
        [np.random.randn(kernel_size, kernel_size) * np.sqrt(2 / (kernel_size * kernel_size))
         for _ in range(out_channels)]
        for _ in range(in_channels)
    ])
    in_channels = out_channels

in_channels_bbox = 1
for l in range(bboxnum_layers):
    out_channels_bbox = bboxnum_kernels * (bboxnum_kernels ** l)
    bbox_kernels.append([
        [np.random.randn(kernel_size, kernel_size) *bboxlearning_rate* np.sqrt(2 / (kernel_size * kernel_size))
         for _ in range(out_channels_bbox)]
        for _ in range(in_channels_bbox)
    ])
    in_channels_bbox = out_channels_bbox


bbbbbbb = {
    "params": None,
}

def init_bbox_head(input_dim):
    hidden_dim = max(16, int(np.sqrt(input_dim)))
    return {
        "W1": np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim),
        "b1": np.zeros((hidden_dim, 1)),
        "W2": np.random.randn(4, hidden_dim) * np.sqrt(2 / hidden_dim),
        "b2": np.zeros((4, 1)),
    }

def bbox_forward(x, input_shape):
    x = x.reshape(-1, 1)
    input_dim = x.shape[0]
    if bbbbbbb["params"] is None:
        bbbbbbb["params"] = init_bbox_head(input_dim)
    p = bbbbbbb["params"]
    z1 = p["W1"] @ x + p["b1"]
    a1 = np.maximum(0, z1)
    z2 = p["W2"] @ a1 + p["b2"]
    out = z2.flatten()
    bbbbbbb["cache"] = {
        "x": x,
        "z1": z1,
        "a1": a1,
        "input_shape": input_shape
    }
    return out

"""
cachebatc=[]
gam = [ 
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

def randbg(fg_img, canv):
    h, w = canv, canv
    c, b = fg_img.shape
    bg = np.random.randint(150, 170, size=(h, w), dtype=np.uint8)
    y = np.random.randint(0, h - c)
    x = np.random.randint(0, w - b)
    bg[y:y+c, x:x+b] = fg_img
    bbox = [x + b/2, y + c/2, b, c]

    return bg, bbox

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

def clip_gradients(g, clip_value=5.0):
    return np.clip(g, -clip_value, clip_value)

def bbox_loss(pred, target, beta=2):
    diff = pred - target
    abs_diff = np.abs(diff)
    loss = np.where(abs_diff < beta,
                    0.5 * (diff ** 2) / beta,
                    abs_diff - 0.5 * beta)
    grad = np.where(abs_diff < beta,
                    diff / beta,
                    np.sign(diff))
    return np.mean(loss), grad * len(pred)

def kernel_learn(kernel, inp, dfeature, lr): # there must be a more stabile way
    kh, kw = kernel.shape
    grad = np.zeros_like(kernel)
    dinput = np.zeros_like(inp)
    for i in range(dfeature.shape[0]):
        for j in range(dfeature.shape[1]):
            patch = inp[i:i+kh, j:j+kw]
            grad += patch * dfeature[i, j]
            dinput[i:i+kh, j:j+kw] += kernel * dfeature[i, j]
    kernel -= lr * grad
    return dinput

def forward_pass_bbox(img): # adjust other methods to take bbox as wel maybe?
    inputs = [[img]]
    for l in range(bboxnum_layers):
        prev_maps = inputs[-1]
        in_channels = len(prev_maps)
        out_channels = len(bbox_kernels[l][0])
        new_maps = []
        for out_k in range(out_channels):
            acc = None
            for in_k in range(in_channels):
                conv = convolve(prev_maps[in_k], bbox_kernels[l][in_k][out_k])
                if acc is None:
                    acc = conv
                else:
                    acc += conv
            act = relu(acc)
            pooled = maxpool(act, pool_size)
            new_maps.append(pooled)
        inputs.append(new_maps)
    return inputs

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

def backward_passbbox(inputs, doutput, original_img):
    next_grads = doutput
    for l in reversed(range(bboxnum_layers)):
        in_channels = len(inputs[l])
        out_channels = len(inputs[l+1])
        new_grads = [np.zeros_like(inputs[l][i]) for i in range(in_channels)]
        for out_k in range(out_channels):
            out_fmap = inputs[l+1][out_k]
            act_fmap = relu(out_fmap)
            dpool = maxpool_backward(next_grads[out_k], act_fmap, pool_size)
            dact = derivrelu(out_fmap) * dpool
            for in_k in range(in_channels):
                dinput = kernel_learn(bbox_kernels[l][in_k][out_k], inputs[l][in_k], dact, bboxlearning_rate)
                new_grads[in_k] += dinput
        next_grads = new_grads
    return next_grads


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
            dact = derivrelu(out_fmap) * dpool
            for in_k in range(in_channels):
                dinput = kernel_learn(kernels[l][in_k][out_k], inputs[l][in_k], dact, learning_rate)
                new_grads[in_k] += dinput
        next_grads = new_grads

def iouuuu(pred, gt):
    px1 = pred[0] - pred[2] / 2
    py1 = pred[1] - pred[3] / 2
    px2 = pred[0] + pred[2] / 2
    py2 = pred[1] + pred[3] / 2
    gx1 = gt[0] - gt[2] / 2
    gy1 = gt[1] - gt[3] / 2
    gx2 = gt[0] + gt[2] / 2
    gy2 = gt[1] + gt[3] / 2
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - intersection

    if union == 0:
        return 0
    return intersection / union

def bbox_backward(dloss_dout, lr=bboxlearning_rate):
    weight_decay = 1e-4
    params = bbbbbbb["params"]
    cache = bbbbbbb["cache"]
    x, z1, a1 = cache["x"], cache["z1"], cache["a1"]
    dZ2 = dloss_dout.reshape(4, 1)
    dW2 = dZ2 @ a1.T
    dW2 = clip_gradients(dW2)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    db2 = clip_gradients(db2)
    dA1 = params["W2"].T @ dZ2
    dA1 = clip_gradients(dA1)
    dZ1 = dA1 * (z1 > 0)
    dW1 = dZ1 @ x.T
    dW1 = clip_gradients(dW1)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    db1 = clip_gradients(db1)
    grad_input = (params["W1"].T @ dZ1).flatten()
    params["W1"] -= lr * (dW1 + weight_decay * params["W1"])
    params["b1"] -= lr * db1
    params["W2"] -= lr * (dW2 + weight_decay * params["W2"])
    params["b2"] -= lr * db2
    return grad_input.reshape(cache["input_shape"])

dumdum = np.zeros((image_size, image_size))
output = forward_pass(dumdum)
final_maps = np.stack(output[-1])
densein = final_maps.size
weights = np.random.randn(len(labeldict), densein) * np.sqrt(2 / densein) * learning_rate # im tired, this is a bad way ik but its the only one i got rn, FIX LATER
final_maps = 0

canv = 120

for epoch in range(epochnum): # huge mess, clean it up
    if (epoch == 0):
        prev_epoch_acc = 0
    errors = []
    iou = []
    files = os.listdir(input_folder)
    for idx, file in enumerate(files):
        #init simplifying kernel here
        img = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE).astype(float)
        bgbg, bbox = randbg(img, canv)
        bgbg = bgbg.astype(np.float32)
        bgbg = bgbg / 255.0
        bbox[0] = bbox[0] / bgbg.shape[1]
        bbox[1] = bbox[1] / bgbg.shape[0]
        bbox[2] = bbox[2] / bgbg.shape[1]
        bbox[3] = bbox[3] / bgbg.shape[0]
        key = cv2.waitKey(1)
        label_file = file.split(".")[0] + ".txt"
        with open(os.path.join(label_folder, label_file)) as f:
            label = labeldict[f.read().strip()]
        bboxlayers = forward_pass_bbox(bgbg)
        bboxmaps = np.stack(bboxlayers[-1])
        bflat = bboxmaps.flatten()
        bbpred = bbox_forward(bflat, bboxmaps.shape)
        bbloss, bbgrad = bbox_loss(bbpred, bbox)
        grad_back = bbox_backward(bbgrad)
        backward_passbbox(bboxlayers, grad_back, bgbg)
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
        bbpred[0] = int(bbpred[0] * bgbg.shape[1])
        bbpred[1] = int(bbpred[1] * bgbg.shape[0])
        bbpred[2] = int(bbpred[2] * bgbg.shape[1])
        bbpred[3] = int(bbpred[3] * bgbg.shape[0])
        bbox[0] = bbox[0] * bgbg.shape[1]
        bbox[1] = bbox[1] * bgbg.shape[0]
        bbox[2] = bbox[2] * bgbg.shape[1]
        bbox[3] = bbox[3] * bgbg.shape[0]
        if np.argmax(logits) != label:
            errors.append(1)
        else:
            errors.append(0)
        iou.append(iouuuu(bbpred, bbox))
        if idx and errors:
            sys.stdout.write("\033[s")
            print(bbloss)
            print(f"%{(np.sum(iou)/idx*100)}")
            print(f"guess: {np.argmax(logits) == label} | 100 acc: %{100-np.sum(errors[-100:])} | total acc: %{100 - (np.sum(errors)/(idx+1)*100)} | loss: {loss}")
            print(f"progress: {idx}/{len(files)} | %{(idx)/len(files)*100}")
            sys.stdout.write("\033[u")
        if(100-np.sum(errors[-100:])<(100-(np.sum(errors)/(idx+1)*100)) and idx % 100):
            #learning_rate*=0.99999 if you want continous learning rate adjustment but really not needed that much
            0
        else:
            0
            #learning_rate*=1.001

    print("\n\n")
    print(f"epoch {epoch+1} accuracy: %{100 - (np.sum(errors) / len(files)) * 100}\n")
    if(prev_epoch_acc > np.sum(errors)/len(files)):
        learning_rate *=0.80
    else:
        learning_rate *=1.20
        with open('kernelsbest.pickle', 'wb') as handle:
            pickle.dump(kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('weightsbest.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    prev_epoch_acc = (np.sum(errors)/len(files))
with open('kernelslast.pickle', 'wb') as handle:
    pickle.dump(kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('weightslast.pickle', 'wb') as handle:
    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bb.pickle', 'wb') as handle:
    pickle.dump(bbbbbbb["params"], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bbkernel.pickle', 'wb') as handle:
    pickle.dump(bbox_kernels, handle, protocol=pickle.HIGHEST_PROTOCOL)
