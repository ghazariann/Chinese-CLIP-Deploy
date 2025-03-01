# t2iRetrieval

    $ conda activate torch2.0
    # column permutation
    $ python t2i_retrieval.py --resume data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt
    # no column permutation
    $ python t2i_retrieval.py --resume data/pretrained_weights/clip_cn_vit-b-16.pt

Then you can see "Input a text (input exit to exit):" in console. It is a retrieval task on Chinese shopping website, so I provide some suitable texts:

    $ 连衣裙 (dress)
    $ 雪地靴 (snow boots)
    $ 运动鞋 (sports shoes)

The amount of retrieval texts is not limited. So you can try to input more texts. And finally you input a "exit", and the retrieval begins. Your retrieavls will be stored in "data/user_input_data/valid_texts.jsonl".

We will get a "valid_texts.txt_feat.jsonl" in transfer_station folder, which is the features of our retrieval texts. We will get a "valid_predictions.jsonl" in data/output_user_data folder, which is the retrieval results. And We will get some folders like "text-1", "text-2" in data/output_user_data folder. The 1, 2, 3 is the same as your input order. In these folders, we can get the retrievals images we want. And their names ""top-1", "top-2", "top-3" are the top-1, top-2, top-3 related retrievals. 

By the way, the images and image features are in the database folder, to simulate it has already been sent from database to TEE.

## The division of the code

### t2i_retrieval.py

line 209-220

### cn_clip/clip/modeling_bert.py

line 515-583


# i2iRetrieval

    $ conda activate torch2.0
    # column permutation
    $ python t2i_retrieval.py --resume data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt
    # no column permutation
    $ python t2i_retrieval.py --resume data/pretrained_weights/clip_cn_vit-b-16.pt

Then you can see "Input a text (input exit to exit):" in console. I don't know how to imitate that a user input an image, so I construct a query_iamges folder to store the query images. And user can input the name of the image in the query_images folder to input an image. I name the images a.jpg to j.jpg, so you can input a to j. I think there is a better way to input an image, a more user-friendly one. But I don't know how to do it.

The amount of retrieval images is not limited. So you can try to input more. And finally you input a "exit", and the retrieval begins. Your retrieavls will be stored as "data/user_input_data/iamge_{name}".

We will get a "query_image_feat.jsonl" in transfer_station folder, which is the features of our retrieval images. We will get a "predictions.jsonl" in data/output_user_data folder, which is the retrieval results. And We will get some folders like "query_image_{name}" in data/output_user_data folder. The name is the same as your input image. In these folders, we can get the retrievals images we want. And their names ""top-1", "top-2", "top-3" are the top-1, top-2, top-3 related retrievals. 

By the way, the images and image features are in the database folder, to simulate it has already been sent from database to TEE. I use test part to be the queried images, because if I use valid part, which the query iamges belong to, the retrieval results will be the same as the query images.

## The division of the code

### i2i_retrieval.py

line 260-265

### cn_clip/clip/model.py

line 333-371
