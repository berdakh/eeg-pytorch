# Motor Imagery EEG data analysis in Pytorch

This repo contains a source code analysis of motor-imagery EEG data in MNE-Python environment, and deep learning model training. 

In particular, on can perform CNN model selection for decoding EEG motor-imagery patterns via PyTorch. This repo was created to help students to get started with EEG and Deep learning research. 

Required packages:
- pip install mne==0.18.0 
- conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
- pip install braindecode


***
# Dataset Information

### Note the datasets and their descriptions were accessed via Mother of All BCI Benchmarks (MOABB)

http://moabb.neurotechx.com/docs/datasets.html

For more details refer to MOABB website, where you can download these datasets. 
 
----- 
## Dataset: 


Four different datasets are used - BCI competition that is publicly available online
 
- 'BNCI2014001R.pickle',
- 'BNCI2014004R.pickle',
- 'Weibo2014R.pickle',
- 'PhysionetRR.pickle'             
 
-------
## BNCI2014001
"""BNCI 2014-001 Motor Imagery dataset.

Dataset IIa from BCI Competition 4 [1]_.

**Dataset Description**

This data set consists of EEG data from 9 subjects.  The cue-based BCI
paradigm consisted of four different motor imagery tasks, namely the imag-
ination of movement of the left hand (class 1), right hand (class 2), both
feet (class 3), and tongue (class 4).  Two sessions on different days were
recorded for each subject.  Each session is comprised of 6 runs separated
by short breaks.  One run consists of 48 trials (12 for each of the four
possible classes), yielding a total of 288 trials per session.

The subjects were sitting in a comfortable armchair in front of a computer
screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
on the black screen.  In addition, a short acoustic warning tone was
presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
pointing either to the left, right, down or up (corresponding to one of the
four classes left hand, right hand, foot or tongue) appeared and stayed on
the screen for 1.25 s.  This prompted the subjects to perform the desired
motor imagery task.  No feedback was provided.  The subjects were ask to
carry out the motor imagery task until the fixation cross disappeared from
the screen at t = 6 s.

Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
were used to record the EEG; the montage is shown in Figure 3 left.  All
signals were recorded monopolarly with the left mastoid serving as
reference and the right mastoid as ground. The signals were sampled with.
250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
the amplifier was set to 100 μV . An additional 50 Hz notch filter was
enabled to suppress line noise

References
----------

.. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
       Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
       and Nolte, G., 2012. Review of the BCI competition IV.
       Frontiers in neuroscience, 6, p.55.
"""

------------
## BNCI2014004
"""BNCI 2014-004 Motor Imagery dataset.

Dataset B from BCI Competition 2008.

**Dataset description**

This data set consists of EEG data from 9 subjects of a study published in
[1]_. The subjects were right-handed, had normal or corrected-to-normal
vision and were paid for participating in the experiments.
All volunteers were sitting in an armchair, watching a flat screen monitor
placed approximately 1 m away at eye level. For each subject 5 sessions
are provided, whereby the first two sessions contain training data without
feedback (screening), and the last three sessions were recorded with
feedback.

Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
and a notch filter at 50 Hz was enabled.  The placement of the three
bipolar recordings (large or small distances, more anterior or posterior)
were slightly different for each subject (for more details see [1]).
The electrode position Fz served as EEG ground. In addition to the EEG
channels, the electrooculogram (EOG) was recorded with three monopolar
electrodes.

The cue-based screening paradigm consisted of two classes,
namely the motor imagery (MI) of left hand (class 1) and right hand
(class 2).
Each subject participated in two screening sessions without feedback
recorded on two different days within two weeks.
Each session consisted of six runs with ten trials each and two classes of
imagery.  This resulted in 20 trials per run and 120 trials per session.

Data of 120 repetitions of each MI class were available for each person in
total.  Prior to the first motor im- agery training the subject executed
and imagined different movements for each body part and selected the one
which they could imagine best (e. g., squeezing a ball or pulling a brake).

Each trial started with a fixation cross and an additional short acoustic
warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
hand movement over a period of 4 seconds.  Each trial was followed by a
short break of at least 1.5 seconds.  A randomized time of up to 1 second
was added to the break to avoid adaptation

For the three online feedback sessions four runs with smiley feedback
were recorded, whereby each run consisted of twenty trials for each type of
motor imagery.  At the beginning of each trial (second 0) the feedback (a
gray smiley) was centered on the screen.  At second 2, a short warning beep
(1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
second 7.5 the screen went blank and a random interval between 1.0 and 2.0
seconds was added to the trial.

References
----------

.. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
       G. Pfurtscheller. Brain-computer communication: motivation, aim,
       and impact of exploring a virtual apartment. IEEE Transactions on
       Neural Systems and Rehabilitation Engineering 15, 473–482, 2007

 
-----------
## Weibo2014 
"""Motor Imagery dataset from Weibo et al 2014.

Dataset from the article *Evaluation of EEG oscillatory patterns and
cognitive process during simple and compound limb motor imagery* [1]_.

It contains data recorded on 10 subjects, with 60 electrodes.

This dataset was used to investigate the differences of the EEG patterns
between simple limb motor imagery and compound limb motor
imagery. Seven kinds of mental tasks have been designed, involving three
tasks of simple limb motor imagery (left hand, right hand, feet), three
tasks of compound limb motor imagery combining hand with hand/foot
(both hands, left hand combined with right foot, right hand combined with
left foot) and rest state.

At the beginning of each trial (8 seconds), a white circle appeared at the
center of the monitor. After 2 seconds, a red circle (preparation cue)
appeared for 1 second to remind the subjects of paying attention to the
character indication next. Then red circle disappeared and character
indication (‘Left Hand’, ‘Left Hand & Right Foot’, et al) was presented on
the screen for 4 seconds, during which the participants were asked to
perform kinesthetic motor imagery rather than a visual type of imagery
while avoiding any muscle movement. After 7 seconds, ‘Rest’ was presented
for 1 second before next trial (Fig. 1(a)). The experiments were divided
into 9 sections, involving 8 sections consisting of 60 trials each for six
kinds of MI tasks (10 trials for each MI task in one section) and one
section consisting of 80 trials for rest state. The sequence of six MI
tasks was randomized. Intersection break was about 5 to 10 minutes.

References
-----------
.. [1] Yi, Weibo, et al. "Evaluation of EEG oscillatory patterns and
       cognitive process during simple and compound limb motor imagery."
       PloS one 9.12 (2014). https://doi.org/10.1371/journal.pone.0114853
"""   

 
#%% PhysionetMI 

"""Physionet Motor Imagery dataset.

   Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

   This data set consists of over 1500 one- and two-minute EEG recordings,
   obtained from 109 volunteers.

   Subjects performed different motor/imagery tasks while 64-channel EEG were
   recorded using the BCI2000 system (http://www.bci2000.org).
   Each subject performed 14 experimental runs: two one-minute baseline runs
   (one with eyes open, one with eyes closed), and three two-minute runs of
   each of the four following tasks:

   1. A target appears on either the left or the right side of the screen.
      The subject opens and closes the corresponding fist until the target
      disappears. Then the subject relaxes.

   2. A target appears on either the left or the right side of the screen.
      The subject imagines opening and closing the corresponding fist until
      the target disappears. Then the subject relaxes.

   3. A target appears on either the top or the bottom of the screen.
      The subject opens and closes either both fists (if the target is on top)
      or both feet (if the target is on the bottom) until the target
      disappears. Then the subject relaxes.

   4. A target appears on either the top or the bottom of the screen.
      The subject imagines opening and closing either both fists
      (if the target is on top) or both feet (if the target is on the bottom)
      until the target disappears. Then the subject relaxes.

   parameters
   ----------

   imagined: bool (default True)
       if True, return runs corresponding to motor imagination.

   executed: bool (default False)
       if True, return runs corresponding to motor execution.

   references
   ----------

   .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
          Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
          interface (BCI) system. IEEE Transactions on biomedical engineering,
          51(6), pp.1034-1043.

   .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
          P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
          H.E. and PhysioBank, P., PhysioNet: components of a new research
          resource for complex physiologic signals Circulation 2000 Volume
          101 Issue 23 pp. E215–E220.
   """
    
    
    
