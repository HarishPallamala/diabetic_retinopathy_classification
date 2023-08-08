clear all;
close all;
clc;
% Create a GUI window
f = figure('Name','Image Classifier','NumberTitle','off','Position',[200 200 500 500]);
% Add a button to load an image
load_button = uicontrol('Style','pushbutton','String','Load Image','Position',[50 450 100 30],'Callback',@load_image);
% Add an axes to display the image
ax = axes('Units','pixels','Position',[50 50 400 400]);
% Add a text label to display the classification result
result_label = uicontrol('Style','text','Position',[200 470 250 20],'HorizontalAlignment','left');
% Load the pre-trained network
load net.mat
% Define the callback function for the load button
function load_image(~,~)
    % Open a file dialog to select an image
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp','Image Files'},'Select an image');
    if isequal(filename,0)
        disp('User selected Cancel')
    else
        % Read the selected image
        %global I
        I = imread(fullfile(pathname,filename));
        I = imresize(I,[336 448]); %as mentioned in paper
        imshow(I);
        
        if size(I,3) == 3
            I = vesselSegPC(I);
        end
    
        % Resize the image to [336 448]
        I2 = imresize(I, [336 448]);
        
        load net.mat;
        % Classify the image using the pre-trained network
        %global net
        [Pre,scores] = classify(net,I2);
        scores = max(double(scores*100));
        % Display the original image with the classification result
        title(join([string(Pre),'',scores,'%']))
        % Display the classification result in the text label
        global result_label
        set(result_label,'String',join([string(Pre),'',scores,'%']))
    end
end
