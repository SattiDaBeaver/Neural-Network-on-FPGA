onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label CLOCK_50 -radix binary /testbench/CLOCK_50
add wave -noupdate -label Reset -radix binary /testbench/reset

add wave -noupdate -divider Inputs
add wave -noupdate -label layerIn -radix hexadecimal /testbench/layerIn
add wave -noupdate -label layerValid -radix hexadecimal /testbench/layerValid

add wave -noupdate -divider "Internal Nets"
add wave -noupdate -label counterWidth -radix hexadecimal /testbench/U1/counterWidth
add wave -noupdate -label serializerOut -radix hexadecimal /testbench/U1/serializerOut

add wave -noupdate -divider "Neuron"
add wave -noupdate -label "FSM State N0" -radix hexadecimal /testbench/U1/gen_neurons\[0\]/Neuron/state
add wave -noupdate -label neuronIn_0 -radix hexadecimal /testbench/U1/gen_neurons\[0\]/Neuron/neuronIn
add wave -noupdate -label neuronValid_0 -radix hexadecimal /testbench/U1/gen_neurons\[0\]/Neuron/neuronValid
add wave -noupdate -label sumOut_0 -radix hexadecimal /testbench/U1/gen_neurons\[0\]/Neuron/sumOut
add wave -noupdate -label neuronOut_0 -radix hexadecimal /testbench/U1/gen_neurons\[0\]/Neuron/neuronOut

add wave -noupdate -divider "Serializer"
add wave -noupdate -label enable -radix hexadecimal /testbench/U1/serializer/enable
add wave -noupdate -label delay -radix hexadecimal /testbench/U1/serializer/delay
add wave -noupdate -label counterOut -radix hexadecimal /testbench/U1/serializer/counterOut
add wave -noupdate -label counterValid -radix hexadecimal /testbench/U1/serializer/counterValid
add wave -noupdate -label serializerOut -radix hexadecimal /testbench/U1/serializer/serializerOut

add wave -noupdate -divider Outputs
add wave -noupdate -label layerOut -radix hexadecimal /testbench/layerOut
add wave -noupdate -label layerOutValid -radix hexadecimal /testbench/layerOutValid


TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {100 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 80
configure wave -valuecolwidth 40
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ps} {1000 ns}
