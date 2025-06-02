# Re-org repo

## step 1: Modularize Transformer - CALCE

- move model part to model.py
- move helper function to utils.py
- add Rated_Capacity and K to train input

## step 2: Generalize code to use on other format of cell data

- get_train_test function should take data frame as input




## Note:

get_train_test() input window_size is the feature_size from train() function. 



如何对比 linear regression 和transformer？
用cross validation 来对比，
使用一种方法，用80%的数据训练，20%的数据验证，算R2，然后重复4次，把每个20%的验证数据平均一下，就可以比较了


使用了传统的transformer来替换paper的Net，其中需要做的变化是把paper里的feature size当成input_dim给传禁区。开始训练后，发现loss会有一些下降，但是非常非常的慢。接下来的尝试：

使用input_dim为1试试，不知道paper里重复了K次的数据是为了什么？这个是loss减少的关键么？