# L46_sim2Real

*The updated final report that includes an abstract is in now uploaded!*

The Github repository is set up in the following way:

<img width="447" alt="Screenshot 2024-01-17 at 7 16 37 PM" src="https://github.com/gkim65/L46_sim2Real/assets/40607001/cd287332-464f-49ac-91e4-c8ff841bdd30">


**Comment 1** - Arduino Code that runs forever reading sensor inputs and sending to the Raspberry Pi through Serial Connection  
**Comment 2** - DQN RL policies trained from pytorch that were used to run on the raspberry pi  
**Comment 3** - Main file used to run all of the DQN RL training, sends data into neptune repository (may need to add in a database api key to run this locally) If wanting to just run the training code, the cartpole_RL.ipynb should be able to run for the most part.  
**Comment 4** - Plots found in paper can be found here as well.  
**Comment 5** - Please install libraries needed to run scripts.  
**Comment 6** - Code run on raspberry pi to step through arduino data, read in the pytorch DQN RL policy model, and use action from that model to run the motor.  


## Videos of Tests

Videos are included in the folder Videos_Hardware, and can be watched by downloading the video (the Partway_success_180_steps.mp4 is the shortest one, 7 seconds)


## Hardware Photos

Photos of the Hardware are included in the folder named Photos_Hardware, and three main descriptor photos can be found below:

![Full Setup](https://github.com/gkim65/L46_sim2Real/blob/main/Photos_Hardware/full_setup_withdescription.png)

________

![Pully setup](https://github.com/gkim65/L46_sim2Real/blob/main/Photos_Hardware/pully_withdescription.png)

________

![pendulum closeup](https://github.com/gkim65/L46_sim2Real/blob/main/Photos_Hardware/Pendulum_closeup_withdescription.png)
