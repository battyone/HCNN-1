function im_h_y = hcnn_Matconvnet(im_l_y,im_l_ycbcr,model,scale,use_cascade)
if use_cascade
    model_scale = 2;
else
    model_scale = scale;
end
iter_all = ceil(log(scale)/log(model_scale));
[lh,lw] = size(im_l_y);
weight = model.weight;
bias = model.bias;
layer_num = size(weight,2);
for iter = 1:iter_all
    fprintf('iter:%d\n',iter);
    im_y = single(imresize(im_l_y,model_scale,'bicubic'));
  
% % bang3_2
convfea=vl_nnconv(im_y,weight{1},bias{1},'Pad',1);
convfea1=vl_nnconv(im_y,weight{21},bias{21},'Pad',1);
     convfea2=vl_nnconv(im_y,weight{26},bias{26},'Pad',1);
    for i = 2:20
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',1);
    end

    for i = 22:25
        convfea1 = vl_nnrelu(convfea1);
        convfea1 = vl_nnconv(convfea1,weight{i},bias{i},'Pad',1);
    end
    
       for i = 27:36         
        convfea2 = vl_nnrelu(convfea2);
         convfea2 = vl_nnconv(convfea2,weight{i},bias{i},'Pad',1);
       end    

 im_h_y = convfea + convfea1+convfea2;

    im_l_y = im_h_y;
end
if size(im_h_y,1) > lh * scale
   im_h_y = gather(im_h_y);
   im_h_y = imresize(im_h_y,[lh * scale,lw * scale],'bicubic');
end
end
