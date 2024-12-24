# gpt


```torchrun --standalone --nproc_per_node=1 train_gpt2_ben.py```

```pip freeze -l > requirements.txt```

```source setup.sh```

```screen -dm bash -c 'torchrun --standalone --nproc_per_node=1 train_gpt2.py > log/screen.txt 2>&1'```

```screen -dm bash -c 'torchrun --standalone --nproc_per_node=8 train_gpt2.py > log/screen.txt 2>&1'```

```screen -r```

```screen -dm bash -c 'torchrun --standalone --nproc_per_node=1 train_gpt2_ben.py > log/screen.txt 2>&1'```


## Notes

I suspect that additional noise will make training much worse... imagine a cryptographic notion, where I (the student) cannot distinguish between a correct answer and an incorrect answer, then of course, I cannot learn! Or, perhaps it takes many more samples for me to distinguish, then I will need substantially many more samples to learn, and perhaps this is compounded over time (not sure how the compounding works). So really, if we do introduce additional sources of loss, we should ensure that it is not noisy...

Let's say we introduce some RL notion, then it had better not be noisy. A data point should only "count" if it is "highly likely" to be correct. Perhaps we should only make it count if the model itself is highly confident in its assessment, and also penalize confident assessments somehow (or perhaps we don't need to do this, because it is already penalized via the ground truth training data: if it is confidently wrong, then it pushes down the score of the true correct answer, which pushes up the loss).

How do we distinguish between positive and negative assessments? Well, in a conversation, say I say something normal, and you reply with something completely out of left field, or completely wrong. Perhaps, in response, I should be exacerbated, or not know what to say? There is no longer a right answer in response to your "random response" (is there?). So perhaps if I am very confident in my reply, this is strong positive feedback. If I am not very confident in my reply at all, then there is no feedback (as opposed to strong negative feedback) because I don't want to introduce too much noise into the learning mechanism. 

So, `block_loss = -log(1 - pr[max])` (the lower the confidence, the closer the block_loss is to 0; the higher the confidence, the more negative the effect on loss.).

### Notes

##### 1-noise

The issue is that the block_loss was only slightly normalized (i.e. for each layer, -1 * crossentropy / n_layers), but this is still very noisy, so learning is not very good.