import os
from options.test_options import TestOptions
from models import create_model
import torch
import numpy as np
import tqdm
import pickle
import cv2
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as scipy_rot


def voting(loss_record, state_record, thresh=20, topk=1, iter=-1, category='laptop'):
    state_rank = get_topk_angle(loss_record, state_record,topk=topk,iter=iter)

    def compare_angle(angle1, angle2):

        R1 = scipy_rot.from_euler('yxz', angle1, degrees=True).as_dcm()[:3, :3]
        R2 = scipy_rot.from_euler('yxz', angle2, degrees=True).as_dcm()[:3, :3]
        
        R1 = R1[:3, :3] / np.cbrt(np.linalg.det(R1[:3, :3]))
        R2 = R2[:3, :3] / np.cbrt(np.linalg.det(R2[:3, :3]))

        if category in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
            y = np.array([0, 1, 0])
            y1 = R1 @ y
            y2 = R2 @ y
            rot_error = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
        else:
            R = R1 @ R2.transpose()
            rot_error = np.arccos((np.trace(R) - 1) / 2)
        
        return rot_error * 180 / np.pi

    ids_inliars_best = []
    for index1, state1 in enumerate(state_rank):
        ids_inliars = [index1]
        for index2, state2 in enumerate(state_rank):
            if compare_angle(state1[:3], state2[:3]) <= thresh:
                ids_inliars.append(index2)
        if len(ids_inliars) > len(ids_inliars_best):
            ids_inliars_best = ids_inliars.copy()

    return state_rank[np.array(ids_inliars_best).min(),:]

def get_topk_angle(loss_record,state_record,topk=1,iter=-1):
    recon_error = loss_record[:,iter,:].sum(-1)
    ranking_sample = [r[0] for r in sorted(enumerate(recon_error), key=lambda r: r[1])]
    return state_record[ranking_sample[:topk],iter,:]




if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 1
    opt.serial_batches = True
    opt.no_flip = True
    
    # https://github.com/hughw19/NOCS_CVPR2019/blob/14dbce775c3c7c45bb7b19269bd53d68efb8f73f/detect_eval.py#L172
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    # Rendering parameters
    focal_lengh_render = 70.
    image_size_render = 64

    # Average scales from the synthetic training set CAMERA
    mean_scales = np.array([0.34, 0.21, 0.19, 0.15, 0.46, 0.17])       
    categories = ['bottle','bowl','camera','can','laptop','mug']
        

    output_folder = os.path.join(opt.results_dir,opt.project_name,opt.test_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    models = []
    for cat in categories:

        opt.category = cat
        opt.exp_name = cat

        model = create_model(opt)
        model.setup(opt)
        model.eval()

        models.append(model)

    nocs_list = sorted(os.listdir( os.path.join(opt.dataroot,'nocs_det')))[::opt.skip]
   
    interval = len(nocs_list)//(opt.num_agent-1) if opt.num_agent > 1 else len(nocs_list)
    task_range = nocs_list[interval*opt.id_agent:min(interval*(opt.id_agent+1), len(nocs_list))]
    
    for file_name in tqdm.tqdm(task_range):
        
        file_path = os.path.join(opt.dataroot,'nocs_det', file_name)
        pose_file = pickle.load(open(file_path, 'rb'), encoding='utf-8')
        
        image_name = pose_file['image_path'].replace('data/real/test', opt.dataroot+'/real_test/')+'_color.png'
        image = cv2.imread(image_name)[:,:,::-1]
        
        masks = pose_file['pred_mask']
        bboxes = pose_file['pred_bboxes']

        pose_file['pred_RTs_ours'] = np.zeros_like(pose_file['pred_RTs'])

        for id, class_pred in enumerate(pose_file['pred_class_ids']):
            bbox = bboxes[id]
            image_mask = image.copy()
            image_mask[masks[:,:,id]==0,:] = 255
            image_mask = image_mask[bbox[0]:bbox[2],bbox[1]:bbox[3],:]

            A = (torch.from_numpy(image_mask.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2) /255) * 2 - 1

            _, c, h, w = A.shape
            s = max( h, w) + 30
            A = F.pad(A,[(s - w)//2, (s - w) - (s - w)//2,
                         (s - h)//2, (s - h) - (s - h)//2],value=1)
            A = F.interpolate(A,size=opt.target_size,mode='bilinear')

            state_history, loss_history, image_history = models[class_pred-1].fitting(A)
            if opt.vis: 
                # Use NOCS's prediction as reference for visualizing pose error as the resuls are not matched to GT's order.
                models[class_pred-1].visulize_fitting(A,torch.tensor(pose_file['pred_RTs'][id]).float().unsqueeze(0),state_history,loss_history,image_history)

            states = voting(loss_history,state_history,category=categories[class_pred-1],topk=5,thresh=10)
            pose_file['pred_RTs_ours'][id][:3,:3] = scipy_rot.from_euler('yxz', states[:3], degrees=True).as_dcm()[:3, :3]


            angle = -states[2] / 180 * np.pi
            mat = np.array([[states[5]*np.cos(angle), -states[5]*np.sin(angle), states[5]*states[3]],
                            [states[5]*np.sin(angle),  states[5]*np.cos(angle), states[5]*states[4]],
                            [                      0,                         0,                 1]])

            mat_inv = np.linalg.inv(mat)
            u = (bbox[1] + bbox[3])/2 + mat_inv[0,2]*s/2
            v = (bbox[0] + bbox[2])/2 + mat_inv[1,2]*s/2
            
            z = image_size_render/(s/states[5]) * (intrinsics[0,0]+intrinsics[1,1])/2 /focal_lengh_render * mean_scales[class_pred-1]

            pose_file['pred_RTs_ours'][id][2, 3] = z
            pose_file['pred_RTs_ours'][id][0, 3] = (u - intrinsics[0,2])/intrinsics[0,0]*z
            pose_file['pred_RTs_ours'][id][1, 3] = (v - intrinsics[1,2])/intrinsics[1,1]*z
            pose_file['pred_RTs_ours'][id][3, 3] = 1

        f = open(os.path.join(output_folder,file_name),'wb')
        pickle.dump(pose_file,f,-1)