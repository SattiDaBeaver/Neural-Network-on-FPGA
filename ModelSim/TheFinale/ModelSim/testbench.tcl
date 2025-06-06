# stop any simulation that is currently running
quit -sim

# create the default "work" library
vlib work;

# compile the Verilog source code in the parent folder
vlog -sv ../top.sv
vlog -sv ../inputShiftRegister.sv
vlog -sv ../NeuralNetwork.sv
vlog -sv ../hardmax.sv
vlog -sv ../layer0.sv
vlog -sv ../layer1.sv
vlog -sv ../neuron.sv
vlog -sv ../reLU.sv
vlog -sv ../weights.sv
vlog -sv ../inputSerializer.sv

# compile the Verilog code of the testbench
vlog -sv *.sv

# start the Simulator, including some libraries that may be needed
vsim work.testbench -Lf 220model -Lf altera_mf_ver -Lf verilog

# show waveforms specified in wave.do
do wave.do

# advance the simulation the desired amount of time
run 30000 ns
