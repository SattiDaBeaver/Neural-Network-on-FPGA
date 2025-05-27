onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label CLOCK_50 -radix binary /testbench/CLOCK_50
add wave -noupdate -label Reset -radix binary /testbench/reset

add wave -noupdate -divider Inputs
add wave -noupdate -label neuronIn -radix hexadecimal /testbench/neuronIn
add wave -noupdate -label neuronValid -radix hexadecimal /testbench/neuronValid

add wave -noupdate -divider "Internal Nets"
add wave -noupdate -label "FSM State" -radix hexadecimal /testbench/U1/state
add wave -noupdate -label MACenable -radix hexadecimal /testbench/U1/MACenable
add wave -noupdate -label activationEnable -radix hexadecimal /testbench/U1/activationEnable
add wave -noupdate -label readEnable -radix hexadecimal /testbench/U1/readEn
add wave -noupdate -label weightAddress -radix hexadecimal /testbench/U1/weightAddress
add wave -noupdate -label weightOut -radix hexadecimal /testbench/U1/weightOut
add wave -noupdate -label biasOut -radix hexadecimal /testbench/U1/biasOut
add wave -noupdate -label sumOut -radix hexadecimal /testbench/U1/sumOut
add wave -noupdate -label reluOut -radix hexadecimal /testbench/U1/reluOut

add wave -noupdate -divider Outputs
add wave -noupdate -label neuronOut -radix hexadecimal /testbench/neuronOut
add wave -noupdate -label neuronOutValid -radix hexadecimal /testbench/neuronOutValid

add wave -noupdate -label counterOut -radix hexadecimal /testbench/inputCounterOut
add wave -noupdate -label counterValid -radix hexadecimal /testbench/inputCounterValid

TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {10000 ps} 0}
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
WaveRestoreZoom {0 ps} {300 ns}
