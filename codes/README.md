# Content
1. Place test.json, train.json, and valid.json under 'aligned' folder. 
2. Change the PATH on aggender_dataset.py to the one leading to your 'aligned' folder.
    e.g. PATH = '/home/jupyter/shared/aligned/'
3. Select a model that you want to train, and find its python file:
    1. Modify caffe models
        a. original_output.py (original caffe model)
        b. one_conv_less.py (remove the third convolutional layer from above)
        c. reduced_both.py (remove the third convolutional layer and a linear layer)
        d. gender_minimalist_output.py (only 1 convolutional layer and 1 linear layer)
    2. Joint classification of age and gender
        a. age_gender_split.py (split adience into 16 catecories)
        b. age_gender_output.py (1 age head and 1 gender head)
        c. gender_with_new_age_encode.py (1 gender head and 8 age heads)
    3. Pretrain on CELEBA then ADIENCE
        a. celeba_output.py (output pretrained caffe model on CELEBA)
        b. pretrained_on_adience.py (finetune the above model on ADIENCE)
4. replace the open("....") with the path to your json file on the previously selected python file.
    e.g.
    train_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/train.json")).items()))
    valid_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/valid.json")).items()))
    test_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/test.json")).items()))

5. Find the corresponding command on "my command.txt" and copy it to console.
6. See your report at the .out file. The graphs can be found in the graph folder, and the trained model under its respective folders.

# For CELEBA:

1. Place CELEBA into a folder. Rename the attribute file as attr.txt, the partition file as partition.txt
2. Copy the folder path to celeba_output.py, replace img_folder. e.g. img_folder = f'/home/jupyter/shared/celeba', where inside celeba you can find img_align_celeba, attr.txt and partition.txt.
3. Run "nohup python3 celeba_output.py > celeba_report.out"

#Trained Models
Since github does not allow me upload files > 25 mb, I will place the models in my google drive. Just replace the respective model folders.
