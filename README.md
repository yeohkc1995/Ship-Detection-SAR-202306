# Ship-Detection-SAR

Credits: 
1) https://github.com/eikeschuett/IcebergShipDetection 
2) https://github.com/Alien9427/DSN/tree/master

IcebergShipDetection:
* Obtain SAR images from the Corpernicus Open Access Hub to test model on real world data.



DSN:

* When implementing the DSN on m2 mac, torch.device is set to MPS and num_workers set to 0. Change the values to suit individual needs.
* Generate bounding box data and training data using Dataset Preparation.ipynb
* Model 2 folder contains the trained joint neural net
* SLC data stored in Classes.zip
* Make sure that the path to boundingbox.csv is mentioned under the read_text() function found in SLC_dataset.py

Step 1: Generate 4D signal spe4D via fft based time-frequency analysis, output spe4_min_max values and img_mean std values

```
python data_process.py --slc_root ../Ship_data_2/Classes/ \                 # single look complex data dir
                       --spe4D_root ../Ship_data_2/spe4D/ \       # 4D TF signal dir
                       --win 0.5                                      # hamming window size (propotion of slc_img, 0.5 or 0.25)
```

Step 2: Train CAE model 
```
python train_cae.py --data_file ../Ship_data2/cae_train.txt ../Ship_data_2/cae_val.txt \
                    --data_root ../Ship_data2/spe4D/ ../Ship_data_2/spe4D/ \
                    --catename2label ../Ship_data_2/class_mapping.txt \
                    --save_model_path ../model/slc_cae_ \
                    --pretrained_model ../model/slc_cae_3.pth \
                    --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                    --device 0
```
Step 3: generate spatially aligned frequency features spe3D using trained cae model
```
python mapping_r4_r3.py --data_txt ../Ship_data_2/file_paths_all.txt \
                        --save_dir ../Ship_data_2/spe3D/ \            # spe3D features
                        --spe_dir ../Ship_data_2/Ship_data_2/spe4D/ \
                        --pretrained_model ../model/slc_cae_3.pth \
                        --catefile ../Ship_data_2/class_mapping.txt \
                        --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                        --batchsize 2
```
Step 4: Get spe3D_max and img_feat_max for feature normalisation
```
python data_process.py --spe3D_root ../Ship_data_2/spe3D/
python get_img_feat_max.py --img_root ../Ship_data_2/Classes/ \
                           --data_file ../Ship_data_2/slc_train.txt \
                           --img_mean_std 0.29982 0.07479776 \
                           --catefile ../Ship_data_2/class_mapping.txt \
                           --cate_num 3 \
                           --device 0
```
Step 5: Train deep network 3 (Use pretrained tsx model)
```
python train_joint.py --img_root ../Ship_data_2/Classes/ ../Ship_data_2/Classes/ \
                      --spe_root ../Ship_data_2/spe3D/ ../Ship_data_2/spe3D/ \
                      --data_file ../Ship_data_2/slc_train.txt ../Ship_data_2/slc_val.txt \
                      --spe3D_max 0.18485471606254578 \
                      --img_feat_max 5.859713554382324 \
                      --img_mean_std 0.29982 0.07479776 \
                      --catefile ../Ship_data_2/class_mapping.txt \
                      --img_model ../model/tsx.pth \
                      --save_model_path ../model2/slc_joint_ \
                      --epoch_num 100 \
                      --cate_num 3 \
                      --device 0
```


