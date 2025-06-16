dataset_paths = {
    'test': '/home/k611/data3/lhy/triplanenet/triplanenet-main/dataset_dsa/L-dataset-corrected-best9test',
    'train' : '/home/k611/data3/lhy/triplanenet/triplanenet-main/dataset_dsa/L-dataset-corrected-best9train',
    # 'train': '/home/k611/data3/ljl/DSADatasets/252DSA_one',
    # 'test': '/home/k611/data3/ljl/DSADatasets/252DSA_one',
    # 'train': '/home/k611/data3/ljl/DSADatasets/R-dataset-corrected-train',
    # 'test': '/home/k611/data3/ljl/DSADatasets/hangyi_R',
}

model_paths = {
    # 'eg3d_dsa': '/home/k611/data3/ljl/DSAGAN/pretrained_models/eg3d_dsa_1432.pth',
    # 'eg3d_dsa': '/home/k611/data3/ljl/DSAGAN/pretrained_models/eg3d_dsa_2444.pth',
    # 'eg3d_dsa': '/home/k611/lys/encoder/train-1/backend-design---encoding/converted_model.pth',
    'eg3d_dsa': '/home/k611/data3/ljl/DSAGAN/pretrained_models/rgb_eg3d_2444.pth',
    # 'eg3d_dsa': '/home/k611/data3/ljl/DSAPretrainedModels/eg3d_R_544.pth',
    # 'eg3d_dsa': '/home/disks/sda/ljl/DSAPretrainedModels/eg3d_R_544.pth',
    # 'eg3d_dsa': "/home/disks/sda/ljl/DSAPretrainedModels/network-snapshot-002444.pkl",
    'facenet':'pretrained_models/TripletLoss_ResNet50_params_epoch5.738275457173586_loss0.9969_acc42.pth',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'ir_se100': 'pretrained_models/model_ir_se100.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'discriminator': 'pretrained_models/discriminator.pth'
}
