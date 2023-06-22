# Ship-Detection-SAR

Credits: 
1) https://github.com/eikeschuett/IcebergShipDetection 
2) https://github.com/Alien9427/DSN/tree/master

IcebergShipDetection:
* Obtain SAR images from the Corpernicus Open Access Hub to test model on real world data.



DSN:

* When implementing the DSN on m2 mac, torch.device is set to MPS and num_workers set to 0. Change the values to suit individual needs.
* Generate bounding box data and training data using Dataset Preparation.ipynb


Step 1: Generate 4D signal spe4D via fft based time-frequency analysis, output spe4_min_max values and img_mean std values

```
python data_process.py --slc_root ../data/slc_data/ \                 # single look complex data dir
                       --spe4D_root ../data/slc_spe4D_fft_12/ \       # 4D TF signal dir
                       --win 0.5                                      # hamming window size (propotion of slc_img, 0.5 or 0.25)
```

Step 2: Train CAE model 
```
python train_cae.py --data_file ../data/slc_cae_train_3.txt ../data/slc_cae_val_3.txt \
                    --data_root ../data/slc_spe4D_fft_12/ ../data/slc_spe4D_fft_12/ \
                    --catename2label ../data/slc_catename2label_cate8.txt \
                    --save_model_path ../model/slc_cae_12_ \
                    --pretrained_model ../model/slc_spexy_cae_3.pth \
                    --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                    --device 0
```
Step 3: generate spatially aligned frequency features spe3D using trained cae model
```
python mapping_r4_r3.py --data_txt ../data/slc_cate8_all.txt \
                        --save_dir ../data/slc_spe4D_fft_12_spe3D/ \            # spe3D features
                        --spe_dir ../data/slc_spe4D_fft_12/ \
                        --pretrained_model ../model/slc_spexy_cae_3.pth \
                        --catefile ../data/slc_catename2label_cate8.txt \
                        --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                        --batchsize 2
```
Step 4: Get spe3D_max and img_feat_max for feature normalisation
```
python data_process.py --spe3D_root ../data/slc_spe4D_fft_12_spe3D/

python get_img_feat_max.py --img_root ../data/slc_data/ \
                           --data_file ../data/slc_train_3.txt \
                           --img_mean_std 0.29982 0.07479776 \
                           --catefile ../data/slc_catename2label_cate8.txt \
                           --cate_num 8 \
                           --device 0
```
Step 5: Train deep network 3 (Use pretrained tsx model)
```
python train_joint.py --img_root ../data/slc_data/ ../data/slc_data/ \
                      --spe_root ../data/slc_spe4D_fft_12_spe3D/ ../data/slc_spe4D_fft_12_spe3D/ \
                      --data_file ../data/slc_train_3.txt ../data/slc_val_3.txt \
                      --spe3D_max 0.18485471606254578 \
                      --img_feat_max 5.859713554382324 \
                      --img_mean_std 0.29982 0.07479776 \
                      --catefile ../data/slc_catename2label_cate8.txt \
                      --img_model ../model/tsx.pth \
                      --save_model_path ../model/slc_joint_ \
                      --epoch_num 100 \
                      --cate_num 8 \
                      --device 0
```


