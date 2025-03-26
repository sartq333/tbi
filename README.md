# tbi: text behind image

Add text in the image's background (something like this, test it out [here](https://huggingface.co/spaces/Sartc/tbi)):

![final_output_3](https://github.com/user-attachments/assets/041a3261-d740-4d3b-b92c-dc28542529c3)

![final_output](https://github.com/user-attachments/assets/ec35957e-b8fc-46fc-9665-eb22e9829628)

![final_output_2](https://github.com/user-attachments/assets/b13eecda-824e-4168-b29e-d3e3132847b9)

This is the [dataset](https://saliencydetection.net/duts/) used for this project. You can also download the dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset).

# Setup details:

Either you can set this up using pip/conda or via [uv](https://x.com/NielsRogge/status/1901210265049342292). I've added the requirements.txt file for this purpose.  

If you want to use it directly locally without training then download the model weights (unet_model.pth) from [here](https://huggingface.co/spaces/Sartc/tbi/tree/main). 

Clone the repository using this command: `git clone https://github.com/sartq333/tbi.git`.

Then create a new conda environment or virtual environment and run this command: `pip install -r requirements.txt`.

After this run this command in your terminal: `python3 app.py`. 

# Work to do:

1. fix alignment issues.
   
~2. host on a gradio UI.~

~3. add details on how the project was done and its setup instructions.~

~4. upload model weights on hf.~

5. compare results of tbi with [rembg](https://github.com/danielgatis/rembg). 

6. give option of different colors for text (fix this on hf spaces, works locally).

7. options for different fonts.
