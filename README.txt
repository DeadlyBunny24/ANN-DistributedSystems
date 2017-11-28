To test:
  - pip install tensorflow
  - python main.py --ms 12702 --hs 200 --lr 0.01 --ep 5

Command line args:
           main.py --ms <memory_size> 
                   --hs <hidden_size>   
                   --lr <learning_rate> 
                   --ep <epochs_of_training> 
                   --help

The model will print the loss for number_files*epochs training iterations. Afterwards, it will be saved on the /tmp folder.

Note: The memory size of the model should be 12702 for this test to work. As this is the 
size of the traces in the traces folder. If you change the size of the traces be sure to use to set the parameter ms to the size of the traces. All traces must be the same size.