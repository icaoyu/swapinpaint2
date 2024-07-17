from PIL import Image
import matplotlib.pyplot as plt
import os

# Log images
def log_input_image(x):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig



def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['att_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])    
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['att_face'])
	plt.title('Att')
    
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')
    
def vis_facesv0(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		vis_faces_w0(hooks_dict, fig, gs, i)
		
	plt.tight_layout()
	return fig
    
def vis_faces_w0(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')

    
    
    
def parse_and_log_images_v2(titlelist,imglist,subdir, title,subscript=None,step=0,display_count=2):
    
#     None, occimg,a_out, id_img, output,
#                                       title='images/test/faces',
#                                       subscript='{:04d}'.format(batch_idx)
    assert len(titlelist)==len(imglist),"length of titlelist is not equal to length of imglist"
    listlen = len(titlelist)
    im_data = []
    for i in range(display_count):
        cur_im_data = {}
        for j in range(listlen):
            cur_im_data[titlelist[j]] = log_input_image(imglist[j][i])
        im_data.append(cur_im_data)
#     save_name = 'inpainted' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.jpg'
    if subscript:
        path = os.path.join(subdir, title, '{}_{:04d}.jpg'.format(subscript, step))
    else:
        path = os.path.join(subdir, title, '{:04d}.jpg'.format(step))
#     path = os.path.join(subdir, title, '{:04d}.jpg'.format(step))
    log_images_v2(im_data,titlelist,path=path)

def log_images_v2(im_data,titlelist,path):
    fig = vis_faces_v2(im_data,titlelist)
    # path = os.path.join(save_dir, '{:04d}.jpg'.format(step))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def vis_faces_v2(log_hooks,titlelist):
    display_count = len(log_hooks)
    titlelen = len(titlelist)
    fig = plt.figure(figsize=(16, 6 * display_count))
    gs = fig.add_gridspec(display_count,titlelen)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        for j in range(titlelen):
            fig.add_subplot(gs[i, j])
            plt.imshow(hooks_dict[titlelist[j]])
            plt.title(titlelist[j])
    plt.tight_layout()
    return fig