# ThesisProcessing
Kyle Weber's MSc Thesis Processing

Programming related to my MSc Kinesiology thesis which involves quantifying activity levels and patterns using
multiple wearable devices (GENEActiv accelerometers on the wrist and ankle, and a chest-worn Bittium Faros ECG). 

Main features of the project:

READING FILES
  -Reading GENEActiv and Bittium Faros files from EDF format
  -Syncing start/end times so that only data from time periods where all devices collected data is included
  
  -Ability to import epoched data from existing .csv files if data has already been processed

EPOCHING DATA
  -Epoching GENEActiv data (activity counts: gravity-subtracted vector magnitudes)
  -Epoching ECG data as average heart rate over a time period

ECG QUALITY CHECK
  -Checking for time periods where ECG data is usable vs. unusable using an adapted algorithm from Orphanidou et al. (2015).
  -Calculates summary measures (% of time invalid, total invalid time, etc.)

SLEEP DATA
  -Reads in data from participant's sleep log
  -Calculates basic metrics (time, % asleep, etc.)
  -Marks epochs as awake, napping, or asleep (overnight)
  
DETERMINATION OF RESTING HEART RATE
  -Definition/method TBD
  
QUANTIFYING ACTIVITY INTENSITY (MODELS)
  -Wrist accelerometer implements the cut-point method from Powell et al. (2016) to estimate if activity was sedentary, 
   light, moderate, or vigorous.
   
  -Ankle accelerometer implements exploratory data processing by calculating a regression equation using activity counts
   during a treadmill protocol where participants walk at 5 speeds to predict gait speed from activity counts. Gait speed 
   is then used to predict VO2, VO2 used to predict METs, and MET levels used to categorize activity (sedentary, light, 
   moderate, vigorous)
   
  -Heart rate is converted to percent heart rate reserve based on derived resting HR and predicted HRmax. %HRR ranges are
   used to categorize intensity into sedentary, light, moderate, or vigorous.
   
   -Summary measures (total minutes, % of data collection) are tallied for each intensity category.

PLOTTING
-Multi-device epoched data: ankle activity counts, wrist activity counts, epoched HR
-Bar plots of total minutes spent at each intensity category as calculated by each of the models
   
