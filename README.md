# ShapeSplitter

Processes a ASCII diagram (A visual shape made from '+', '-', '|', newline characters and spaces) and splits the shape into its most minimal subshapes. If there are subshapes nested (fully enclosed in) other subshapes, then the parent subshape will be drawn with a hole that is the shape of the child subshape.

Input: .txt file with a ASCII diagram. Some example inputs (shape1.txt, shape2.txt, shape3.txt) are provided.

Usage: python shape_split.py input.txt

Output: The minimal subshapes will be printed and also written to a file called input_split.py.

Note: With the example input shape3.txt, the ASCII diagram is quite wide. It is recommended that this file and its corresponding output file be viewed on a text editor that does not have margins.
