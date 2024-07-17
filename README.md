# swapinpaint2

pythorch implementation of SwapInpaint2

step1:

conda create --name new_env --file requirements.txt

step2:pre-trained model, Put it in the chkpt/ folder

链接: https://pan.baidu.com/s/1Q-x-5ucBBUMFJ0LsvnWLGw?pwd=dr3v 提取码: dr3v 


face inpainting:
run
python scripts/inference-ID256-biaad-wbg.py --gt_img data/test_jpg/36.jpg --id_img data/test_jpg/36.jpg --mask_type 0