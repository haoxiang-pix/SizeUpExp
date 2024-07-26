import cv2
import numpy as np
import tqdm
from clothseg.process import process_main as seg_cloth

cloth_seg_model_path = 'clothseg/model/cloth_segm.pth'

# src_path = '/data/lihaox/samples/1.png'
# dst_path = '/data/lihaox/samples/vmake-1.png'

# src_path = '/data/lihaox/samples/sysuupper00534398_00.jpg'
# dst_path = '/data/lihaox/samples/sysuupper00534398_00-vmake.png'

src_path = '/data/lihaox/samples/A_Regular.jpeg'
dst_path = '/data/lihaox/samples/A_Plus.jpeg'

srcI = cv2.imread(src_path)
dstI = cv2.imread(dst_path)
_, src_cloth_mask = seg_cloth(cloth_seg_model_path, src_path, 'cuda')
_, dst_cloth_mask = seg_cloth(cloth_seg_model_path, dst_path, 'cuda')
src_mask = (src_cloth_mask == 1).astype(np.uint8)
dst_mask = (dst_cloth_mask == 1).astype(np.uint8)
src_mask = cv2.erode(src_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
dst_mask = cv2.erode(dst_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

## crop out inputs
def crop_and_resize(I, mask, target_size=512, bg_color=(0, 0, 0)):
    y_acc = np.where(np.max(mask, axis=1) > 0)[0]
    miny, maxy = y_acc[0], y_acc[-1]+1
    x_acc = np.where(np.max(mask, axis=0) > 0)[0]
    minx, maxx = x_acc[0], x_acc[-1]+1
    bbox = [minx, miny, maxx, maxy]

    bbox_wh = max(maxy - miny, maxx - minx)
    bbox_ctr = [(minx + maxx)/2, (miny + maxy)/2]
    bbox = [int(bbox_ctr[0] - bbox_wh/2), int(bbox_ctr[1] - bbox_wh/2), int(bbox_ctr[0] + bbox_wh/2), int(bbox_ctr[1] + bbox_wh/2)]

    padding_color = bg_color #np.mean(np.array([I[0,0,:], I[-1,-1,:], I[0,-1,:], I[-1,0,:]]), axis=0)
    padding_size = max([0, -bbox[0], -bbox[1], bbox[2] - I.shape[1], bbox[3] - I.shape[0]])
    if padding_size > 0:
        I = cv2.copyMakeBorder(I, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=padding_color)
        mask = cv2.copyMakeBorder(mask, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0))
        bbox = [bbox[0]+padding_size, bbox[1]+padding_size, bbox[2]+padding_size, bbox[3]+padding_size]
    content = I[bbox[1]:bbox[3], bbox[0]:bbox[2], :3]
    content_alpha = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]][:,:,None]
    content = (content * (content_alpha) + np.array(bg_color) * (1 - content_alpha)).astype(np.uint8)
    mask = cv2.resize(content_alpha[:, :, 0], (target_size, target_size), cv2.INTER_LANCZOS4)
    return content, cv2.resize(content, (target_size, target_size),cv2.INTER_LANCZOS4), mask

target_size = 512
src_orig, src_crop, src_crop_mask = crop_and_resize(srcI, src_mask)
dst_orig, dst_crop, dst_crop_mask = crop_and_resize(dstI, dst_mask)

cv2.imwrite('tmp/src_crop.png', src_crop)
cv2.imwrite('tmp/dst_crop.png', dst_crop)

from model import FlowGenerator, FlowUNet2DModel
h, w = src_crop_mask.shape[:2]
print (src_crop_mask.shape)
src_data = (src_crop_mask.reshape(1, 1, h, w) - 0.5)*2 # -1 ~ 1
dst_data = (dst_crop_mask.reshape(1, 1, h, w) - 0.5)*2

import torch
weight_dtype = torch.float32
device = 'cuda'
src = torch.from_numpy(src_data).to(device, dtype=weight_dtype)
dst = torch.from_numpy(dst_data).to(device, dtype=weight_dtype)

debug_dir = 'tmp/debug/'

src_crop_ts = torch.from_numpy(src_crop).to(device, dtype=weight_dtype)
dst_crop_ts = torch.from_numpy(dst_crop).to(device, dtype=weight_dtype)
src_crop_ts = (src_crop_ts/255. - 0.5)*2
dst_crop_ts = (dst_crop_ts/255. - 0.5)*2

src, dst = src_crop_ts[None,...].permute(0, 3, 1, 2), dst_crop_ts[None,...].permute(0, 3, 1, 2)
num_channels = src.shape[1] + dst.shape[1]

from torch.nn import functional as F
def warp_func(x, flow, grid, mode='bilinear', padding_mode='zeros', coff=0.2):
    flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
    if grid is None:
        with torch.no_grad():
            n, c, h, w = x.size()
            yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
            xv = xv.float() / (w - 1) * 2.0 - 1
            yv = yv.float() / (h - 1) * 2.0 - 1
            if torch.cuda.is_available():
                grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
            else:
                grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
    grid_x = grid + 2 * flow * coff
    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
    return warp_x, grid

class NaiveFlow(torch.nn.Module):
    def __init__(self, h, w):
        super(NaiveFlow, self).__init__()
        self.flow = torch.nn.Parameter((torch.rand(1, 2, h, w)*2-1)*0.01, requires_grad=True)
    def forward(self, img1, img2, coff):
        flow = self.flow
        warp_x = warp_func(img1, flow, coff=coff)
        warp_x = torch.clamp(warp_x, min=-1.0, max=1.0)
        return warp_x, flow

srcI, dstI = src_crop, dst_crop
# mm = NaiveFlow(srcI.shape[0], srcI.shape[1]).to(device).to(weight_dtype)
# mm = FlowGenerator(num_channels).to(device).to(weight_dtype)
mm = FlowUNet2DModel().to(device).to(weight_dtype)

warp_coeff = 0.5
optimizer = torch.optim.Adam(mm.parameters(), lr=1e-3)
max_step = 5000
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, eta_min=1e-5)
loss_func = torch.nn.MSELoss()
grid = None
for step in tqdm.tqdm(range(max_step)):
    optimizer.zero_grad()
    flow = mm(src)# dst, coff=warp_coeff)
    warp, grid = warp_func(src, flow, grid, coff=warp_coeff)
    loss = loss_func(warp, dst)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    
    if step % 500 == 0:
        print (step, loss.item())
        # warpI, _ = mm(src_crop_ts[None,...].permute(3, 0, 1, 2), dst_crop_ts[None,...].permute(3, 0, 1, 2))
        # warpI = ((warpI.permute(1, 2, 3, 0)[0]+1)/2.*255).detach().cpu().numpy().astype(np.uint8)
        # warpM = ((warp[0]+1)/2.*255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        # visM = np.hstack((src_crop_mask*255, dst_crop_mask*255, warpM[:, :, 0]))
        # visI = np.hstack((src_crop, dst_crop, warpI))
        # visM = cv2.cvtColor(visM, cv2.COLOR_GRAY2BGR)
        # visF = np.vstack((visM, visI))
        warpI = ((warp[0]+1)/2.*255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        visF = np.hstack((srcI, dstI, warpI))
        debug_fpath = f"{debug_dir}/{step:05d}.jpg"
        cv2.imwrite(debug_fpath, visF)

flow_x = flow[0, :, :, 0].detach().cpu().numpy()
flow_y = flow[0, :, :, 1].detach().cpu().numpy()
flow_field_xo = cv2.resize(flow_x, (src_orig.shape[1], src_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
flow_field_yo = cv2.resize(flow_y, (src_orig.shape[1], src_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
flow_field_x = flow_field_xo * src_orig.shape[1] * 0.5 * 2 * warp_coeff
flow_field_y = flow_field_yo * src_orig.shape[0] * 0.5 * 2 * warp_coeff

def vis_flow(flow_field, grid_size=10):
    h, w = flow_field.shape[:2]
    step = grid_size*3
    visI = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(step, h-step, step):
        for x in range(step, w-step, step):
            loc0 = (x, y)
            loc1 = (int(flow_field[y, x]*grid_size + loc0[0]), int(flow_field[y, x]*grid_size + loc0[1]))
            cv2.arrowedLine(visI, loc0, loc1, color=(0, 0, 255), thickness=1, line_type=cv2.LINE_AA, shift=0)
    return visI
cv2.imwrite('tmp/flow_x.png', vis_flow(flow_field_xo))
cv2.imwrite('tmp/flow_y.png', vis_flow(flow_field_yo))

from image_warp import image_warp_grid1
pred_orig  = image_warp_grid1(flow_field_x, flow_field_y, src_orig, 1.0, 0, 0)
cv2.imwrite('tmp/orig_pred.png', pred_orig)
cv2.imwrite('tmp/orig_src.png', src_orig)
cv2.imwrite('tmp/orig_dst.png', dst_orig)
import pdb; pdb.set_trace()
