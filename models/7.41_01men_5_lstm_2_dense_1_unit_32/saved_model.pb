Мм;
┐ј
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceѕ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeіьout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ке7
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
ђ
Adam/v/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_32/bias
y
(Adam/v/dense_32/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_32/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_32/bias
y
(Adam/m/dense_32/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_32/bias*
_output_shapes
:*
dtype0
ѕ
Adam/v/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/v/dense_32/kernel
Ђ
*Adam/v/dense_32/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_32/kernel*
_output_shapes

: *
dtype0
ѕ
Adam/m/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/m/dense_32/kernel
Ђ
*Adam/m/dense_32/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_32/kernel*
_output_shapes

: *
dtype0
ђ
Adam/v/dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_31/bias
y
(Adam/v/dense_31/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_31/bias*
_output_shapes
: *
dtype0
ђ
Adam/m/dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_31/bias
y
(Adam/m/dense_31/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_31/bias*
_output_shapes
: *
dtype0
ѕ
Adam/v/dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_31/kernel
Ђ
*Adam/v/dense_31/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_31/kernel*
_output_shapes

:  *
dtype0
ѕ
Adam/m/dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_31/kernel
Ђ
*Adam/m/dense_31/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_31/kernel*
_output_shapes

:  *
dtype0
Њ
Adam/v/lstm_38/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/v/lstm_38/lstm_cell/bias
ї
1Adam/v/lstm_38/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_38/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
Њ
Adam/m/lstm_38/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/m/lstm_38/lstm_cell/bias
ї
1Adam/m/lstm_38/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_38/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
»
)Adam/v/lstm_38/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/v/lstm_38/lstm_cell/recurrent_kernel
е
=Adam/v/lstm_38/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_38/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
»
)Adam/m/lstm_38/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/m/lstm_38/lstm_cell/recurrent_kernel
е
=Adam/m/lstm_38/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_38/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/v/lstm_38/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/v/lstm_38/lstm_cell/kernel
ћ
3Adam/v/lstm_38/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_38/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/m/lstm_38/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/m/lstm_38/lstm_cell/kernel
ћ
3Adam/m/lstm_38/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_38/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Њ
Adam/v/lstm_37/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/v/lstm_37/lstm_cell/bias
ї
1Adam/v/lstm_37/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_37/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
Њ
Adam/m/lstm_37/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/m/lstm_37/lstm_cell/bias
ї
1Adam/m/lstm_37/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_37/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
»
)Adam/v/lstm_37/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/v/lstm_37/lstm_cell/recurrent_kernel
е
=Adam/v/lstm_37/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_37/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
»
)Adam/m/lstm_37/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/m/lstm_37/lstm_cell/recurrent_kernel
е
=Adam/m/lstm_37/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_37/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/v/lstm_37/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/v/lstm_37/lstm_cell/kernel
ћ
3Adam/v/lstm_37/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_37/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/m/lstm_37/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/m/lstm_37/lstm_cell/kernel
ћ
3Adam/m/lstm_37/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_37/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Њ
Adam/v/lstm_36/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/v/lstm_36/lstm_cell/bias
ї
1Adam/v/lstm_36/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_36/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
Њ
Adam/m/lstm_36/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/m/lstm_36/lstm_cell/bias
ї
1Adam/m/lstm_36/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_36/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
»
)Adam/v/lstm_36/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/v/lstm_36/lstm_cell/recurrent_kernel
е
=Adam/v/lstm_36/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_36/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
»
)Adam/m/lstm_36/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/m/lstm_36/lstm_cell/recurrent_kernel
е
=Adam/m/lstm_36/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_36/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/v/lstm_36/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/v/lstm_36/lstm_cell/kernel
ћ
3Adam/v/lstm_36/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_36/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/m/lstm_36/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*0
shared_name!Adam/m/lstm_36/lstm_cell/kernel
ћ
3Adam/m/lstm_36/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_36/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Њ
Adam/v/lstm_35/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/v/lstm_35/lstm_cell/bias
ї
1Adam/v/lstm_35/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_35/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
Њ
Adam/m/lstm_35/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*.
shared_nameAdam/m/lstm_35/lstm_cell/bias
ї
1Adam/m/lstm_35/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_35/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
»
)Adam/v/lstm_35/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/v/lstm_35/lstm_cell/recurrent_kernel
е
=Adam/v/lstm_35/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_35/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
»
)Adam/m/lstm_35/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*:
shared_name+)Adam/m/lstm_35/lstm_cell/recurrent_kernel
е
=Adam/m/lstm_35/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_35/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Џ
Adam/v/lstm_35/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*0
shared_name!Adam/v/lstm_35/lstm_cell/kernel
ћ
3Adam/v/lstm_35/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_35/lstm_cell/kernel*
_output_shapes
:	ђ*
dtype0
Џ
Adam/m/lstm_35/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*0
shared_name!Adam/m/lstm_35/lstm_cell/kernel
ћ
3Adam/m/lstm_35/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_35/lstm_cell/kernel*
_output_shapes
:	ђ*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Ё
lstm_38/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*'
shared_namelstm_38/lstm_cell/bias
~
*lstm_38/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_38/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
А
"lstm_38/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*3
shared_name$"lstm_38/lstm_cell/recurrent_kernel
џ
6lstm_38/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_38/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Ї
lstm_38/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*)
shared_namelstm_38/lstm_cell/kernel
є
,lstm_38/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_38/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Ё
lstm_37/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*'
shared_namelstm_37/lstm_cell/bias
~
*lstm_37/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_37/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
А
"lstm_37/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*3
shared_name$"lstm_37/lstm_cell/recurrent_kernel
џ
6lstm_37/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_37/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Ї
lstm_37/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*)
shared_namelstm_37/lstm_cell/kernel
є
,lstm_37/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_37/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Ё
lstm_36/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*'
shared_namelstm_36/lstm_cell/bias
~
*lstm_36/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
А
"lstm_36/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*3
shared_name$"lstm_36/lstm_cell/recurrent_kernel
џ
6lstm_36/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_36/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Ї
lstm_36/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*)
shared_namelstm_36/lstm_cell/kernel
є
,lstm_36/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell/kernel*
_output_shapes
:	 ђ*
dtype0
Ё
lstm_35/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*'
shared_namelstm_35/lstm_cell/bias
~
*lstm_35/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
А
"lstm_35/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ђ*3
shared_name$"lstm_35/lstm_cell/recurrent_kernel
џ
6lstm_35/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_35/lstm_cell/recurrent_kernel*
_output_shapes
:	 ђ*
dtype0
Ї
lstm_35/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*)
shared_namelstm_35/lstm_cell/kernel
є
,lstm_35/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

: *
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
: *
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:  *
dtype0
ѕ
serving_default_lstm_35_inputPlaceholder*+
_output_shapes
:         
*
dtype0* 
shape:         

­
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_35_inputlstm_35/lstm_cell/kernel"lstm_35/lstm_cell/recurrent_kernellstm_35/lstm_cell/biaslstm_36/lstm_cell/kernel"lstm_36/lstm_cell/recurrent_kernellstm_36/lstm_cell/biaslstm_37/lstm_cell/kernel"lstm_37/lstm_cell/recurrent_kernellstm_37/lstm_cell/biaslstm_38/lstm_cell/kernel"lstm_38/lstm_cell/recurrent_kernellstm_38/lstm_cell/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_762730

NoOpNoOp
уѕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Аѕ
valueќѕBњѕ Bіѕ
Э
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
Ц
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_random_generator* 
┴
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator
,cell
-
state_spec*
Ц
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator* 
┴
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator
<cell
=
state_spec*
Ц
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
D_random_generator* 
┴
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator
Lcell
M
state_spec*
Ц
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator* 
д
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
Ц
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
c_random_generator* 
д
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias*
z
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11
[12
\13
j14
k15*
z
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11
[12
\13
j14
k15*
* 
░
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

}trace_0
~trace_1* 

trace_0
ђtrace_1* 
* 
ѕ
Ђ
_variables
ѓ_iterations
Ѓ_learning_rate
ё_index_dict
Ё
_momentums
є_velocities
Є_update_step_xla*

ѕserving_default* 

l0
m1
n2*

l0
m1
n2*
* 
Ц
Ѕstates
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Јtrace_0
љtrace_1
Љtrace_2
њtrace_3* 
:
Њtrace_0
ћtrace_1
Ћtrace_2
ќtrace_3* 
* 
в
Ќ	variables
ўtrainable_variables
Ўregularization_losses
џ	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses
Ю_random_generator
ъ
state_size

lkernel
mrecurrent_kernel
nbias*
* 
* 
* 
* 
ќ
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

цtrace_0
Цtrace_1* 

дtrace_0
Дtrace_1* 
* 

o0
p1
q2*

o0
p1
q2*
* 
Ц
еstates
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
:
«trace_0
»trace_1
░trace_2
▒trace_3* 
:
▓trace_0
│trace_1
┤trace_2
хtrace_3* 
* 
в
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses
╝_random_generator
й
state_size

okernel
precurrent_kernel
qbias*
* 
* 
* 
* 
ќ
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

├trace_0
─trace_1* 

┼trace_0
кtrace_1* 
* 

r0
s1
t2*

r0
s1
t2*
* 
Ц
Кstates
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
:
═trace_0
╬trace_1
¤trace_2
лtrace_3* 
:
Лtrace_0
мtrace_1
Мtrace_2
нtrace_3* 
* 
в
Н	variables
оtrainable_variables
Оregularization_losses
п	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses
█_random_generator
▄
state_size

rkernel
srecurrent_kernel
tbias*
* 
* 
* 
* 
ќ
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

Рtrace_0
сtrace_1* 

Сtrace_0
тtrace_1* 
* 

u0
v1
w2*

u0
v1
w2*
* 
Ц
Тstates
уnon_trainable_variables
Уlayers
жmetrics
 Жlayer_regularization_losses
вlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
:
Вtrace_0
ьtrace_1
Ьtrace_2
№trace_3* 
:
­trace_0
ыtrace_1
Ыtrace_2
зtrace_3* 
* 
в
З	variables
шtrainable_variables
Шregularization_losses
э	keras_api
Э__call__
+щ&call_and_return_all_conditional_losses
Щ_random_generator
ч
state_size

ukernel
vrecurrent_kernel
wbias*
* 
* 
* 
* 
ќ
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

Ђtrace_0
ѓtrace_1* 

Ѓtrace_0
ёtrace_1* 
* 

[0
\1*

[0
\1*
* 
ў
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

іtrace_0* 

Іtrace_0* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

Љtrace_0
њtrace_1* 

Њtrace_0
ћtrace_1* 
* 

j0
k1*

j0
k1*
* 
ў
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

џtrace_0* 

Џtrace_0* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_35/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_35/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_35/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_36/lstm_cell/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_36/lstm_cell/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_36/lstm_cell/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_37/lstm_cell/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_37/lstm_cell/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_37/lstm_cell/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_38/lstm_cell/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"lstm_38/lstm_cell/recurrent_kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_38/lstm_cell/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

ю0
Ю1*
* 
* 
* 
* 
* 
* 
Б
ѓ0
ъ1
Ъ2
а3
А4
б5
Б6
ц7
Ц8
д9
Д10
е11
Е12
ф13
Ф14
г15
Г16
«17
»18
░19
▒20
▓21
│22
┤23
х24
Х25
и26
И27
╣28
║29
╗30
╝31
й32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
і
ъ0
а1
б2
ц3
д4
е5
ф6
г7
«8
░9
▓10
┤11
Х12
И13
║14
╝15*
і
Ъ0
А1
Б2
Ц3
Д4
Е5
Ф6
Г7
»8
▒9
│10
х11
и12
╣13
╗14
й15*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

l0
m1
n2*

l0
m1
n2*
* 
ъ
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
Ќ	variables
ўtrainable_variables
Ўregularization_losses
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses*

├trace_0
─trace_1* 

┼trace_0
кtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

o0
p1
q2*

o0
p1
q2*
* 
ъ
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Х	variables
иtrainable_variables
Иregularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses*

╠trace_0
═trace_1* 

╬trace_0
¤trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1
t2*

r0
s1
t2*
* 
ъ
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
Н	variables
оtrainable_variables
Оregularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses*

Нtrace_0
оtrace_1* 

Оtrace_0
пtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

L0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

u0
v1
w2*

u0
v1
w2*
* 
ъ
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
Пlayer_metrics
З	variables
шtrainable_variables
Шregularization_losses
Э__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses*

яtrace_0
▀trace_1* 

Яtrace_0
рtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Р	variables
с	keras_api

Сtotal

тcount*
M
Т	variables
у	keras_api

Уtotal

жcount
Ж
_fn_kwargs*
jd
VARIABLE_VALUEAdam/m/lstm_35/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_35/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/lstm_35/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/lstm_35/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/lstm_35/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/lstm_35/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/lstm_36/lstm_cell/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_36/lstm_cell/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/lstm_36/lstm_cell/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/lstm_36/lstm_cell/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_36/lstm_cell/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_36/lstm_cell/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/lstm_37/lstm_cell/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/lstm_37/lstm_cell/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/lstm_37/lstm_cell/recurrent_kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/lstm_37/lstm_cell/recurrent_kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_37/lstm_cell/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_37/lstm_cell/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/lstm_38/lstm_cell/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/lstm_38/lstm_cell/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/lstm_38/lstm_cell/recurrent_kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/lstm_38/lstm_cell/recurrent_kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_38/lstm_cell/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_38/lstm_cell/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_31/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_31/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_31/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_31/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_32/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_32/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_32/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_32/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

С0
т1*

Р	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

У0
ж1*

Т	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
│
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/biaslstm_35/lstm_cell/kernel"lstm_35/lstm_cell/recurrent_kernellstm_35/lstm_cell/biaslstm_36/lstm_cell/kernel"lstm_36/lstm_cell/recurrent_kernellstm_36/lstm_cell/biaslstm_37/lstm_cell/kernel"lstm_37/lstm_cell/recurrent_kernellstm_37/lstm_cell/biaslstm_38/lstm_cell/kernel"lstm_38/lstm_cell/recurrent_kernellstm_38/lstm_cell/bias	iterationlearning_rateAdam/m/lstm_35/lstm_cell/kernelAdam/v/lstm_35/lstm_cell/kernel)Adam/m/lstm_35/lstm_cell/recurrent_kernel)Adam/v/lstm_35/lstm_cell/recurrent_kernelAdam/m/lstm_35/lstm_cell/biasAdam/v/lstm_35/lstm_cell/biasAdam/m/lstm_36/lstm_cell/kernelAdam/v/lstm_36/lstm_cell/kernel)Adam/m/lstm_36/lstm_cell/recurrent_kernel)Adam/v/lstm_36/lstm_cell/recurrent_kernelAdam/m/lstm_36/lstm_cell/biasAdam/v/lstm_36/lstm_cell/biasAdam/m/lstm_37/lstm_cell/kernelAdam/v/lstm_37/lstm_cell/kernel)Adam/m/lstm_37/lstm_cell/recurrent_kernel)Adam/v/lstm_37/lstm_cell/recurrent_kernelAdam/m/lstm_37/lstm_cell/biasAdam/v/lstm_37/lstm_cell/biasAdam/m/lstm_38/lstm_cell/kernelAdam/v/lstm_38/lstm_cell/kernel)Adam/m/lstm_38/lstm_cell/recurrent_kernel)Adam/v/lstm_38/lstm_cell/recurrent_kernelAdam/m/lstm_38/lstm_cell/biasAdam/v/lstm_38/lstm_cell/biasAdam/m/dense_31/kernelAdam/v/dense_31/kernelAdam/m/dense_31/biasAdam/v/dense_31/biasAdam/m/dense_32/kernelAdam/v/dense_32/kernelAdam/m/dense_32/biasAdam/v/dense_32/biastotal_1count_1totalcountConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_766114
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/biaslstm_35/lstm_cell/kernel"lstm_35/lstm_cell/recurrent_kernellstm_35/lstm_cell/biaslstm_36/lstm_cell/kernel"lstm_36/lstm_cell/recurrent_kernellstm_36/lstm_cell/biaslstm_37/lstm_cell/kernel"lstm_37/lstm_cell/recurrent_kernellstm_37/lstm_cell/biaslstm_38/lstm_cell/kernel"lstm_38/lstm_cell/recurrent_kernellstm_38/lstm_cell/bias	iterationlearning_rateAdam/m/lstm_35/lstm_cell/kernelAdam/v/lstm_35/lstm_cell/kernel)Adam/m/lstm_35/lstm_cell/recurrent_kernel)Adam/v/lstm_35/lstm_cell/recurrent_kernelAdam/m/lstm_35/lstm_cell/biasAdam/v/lstm_35/lstm_cell/biasAdam/m/lstm_36/lstm_cell/kernelAdam/v/lstm_36/lstm_cell/kernel)Adam/m/lstm_36/lstm_cell/recurrent_kernel)Adam/v/lstm_36/lstm_cell/recurrent_kernelAdam/m/lstm_36/lstm_cell/biasAdam/v/lstm_36/lstm_cell/biasAdam/m/lstm_37/lstm_cell/kernelAdam/v/lstm_37/lstm_cell/kernel)Adam/m/lstm_37/lstm_cell/recurrent_kernel)Adam/v/lstm_37/lstm_cell/recurrent_kernelAdam/m/lstm_37/lstm_cell/biasAdam/v/lstm_37/lstm_cell/biasAdam/m/lstm_38/lstm_cell/kernelAdam/v/lstm_38/lstm_cell/kernel)Adam/m/lstm_38/lstm_cell/recurrent_kernel)Adam/v/lstm_38/lstm_cell/recurrent_kernelAdam/m/lstm_38/lstm_cell/biasAdam/v/lstm_38/lstm_cell/biasAdam/m/dense_31/kernelAdam/v/dense_31/kernelAdam/m/dense_31/biasAdam/v/dense_31/biasAdam/m/dense_32/kernelAdam/v/dense_32/kernelAdam/m/dense_32/biasAdam/v/dense_32/biastotal_1count_1totalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_766285яЃ5
М
х
(__inference_lstm_35_layer_call_fn_762774

inputs
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_761959s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762770:&"
 
_user_specified_name762768:&"
 
_user_specified_name762766:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬$
╬
while_body_759800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_759824_0:	ђ+
while_lstm_cell_759826_0:	 ђ'
while_lstm_cell_759828_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_759824:	ђ)
while_lstm_cell_759826:	 ђ%
while_lstm_cell_759828:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_759824_0while_lstm_cell_759826_0while_lstm_cell_759828_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759786┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_759824while_lstm_cell_759824_0"2
while_lstm_cell_759826while_lstm_cell_759826_0"2
while_lstm_cell_759828while_lstm_cell_759828_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name759828:&	"
 
_user_specified_name759826:&"
 
_user_specified_name759824:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И8
щ
C__inference_lstm_36_layer_call_and_return_conditional_losses_760215

inputs#
lstm_cell_760133:	 ђ#
lstm_cell_760135:	 ђ
lstm_cell_760137:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760133lstm_cell_760135lstm_cell_760137*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760132n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760133lstm_cell_760135lstm_cell_760137*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760146*
condR
while_cond_760145*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760137:&"
 
_user_specified_name760135:&"
 
_user_specified_name760133:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
ж
d
F__inference_dropout_56_layer_call_and_return_conditional_losses_762127

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
зJ
І
C__inference_lstm_38_layer_call_and_return_conditional_losses_764848
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764763*
condR
while_cond_764762*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
┬	
├
while_cond_761172
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_761172___redundant_placeholder04
0while_while_cond_761172___redundant_placeholder14
0while_while_cond_761172___redundant_placeholder24
0while_while_cond_761172___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759786

inputs

states
states_11
matmul_readvariableop_resource:	ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765572

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬	
├
while_cond_762832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_762832___redundant_placeholder04
0while_while_cond_762832___redundant_placeholder14
0while_while_cond_762832___redundant_placeholder24
0while_while_cond_762832___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И8
щ
C__inference_lstm_37_layer_call_and_return_conditional_losses_760706

inputs#
lstm_cell_760624:	 ђ#
lstm_cell_760626:	 ђ
lstm_cell_760628:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760624lstm_cell_760626lstm_cell_760628*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760623n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760624lstm_cell_760626lstm_cell_760628*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760637*
condR
while_cond_760636*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760628:&"
 
_user_specified_name760626:&"
 
_user_specified_name760624:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
ж
d
F__inference_dropout_55_layer_call_and_return_conditional_losses_761971

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
ѕJ
І
C__inference_lstm_35_layer_call_and_return_conditional_losses_762917
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_762833*
condR
while_cond_762832*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
є:
х
while_body_765198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ћЦ
Д
!__inference__wrapped_model_759724
lstm_35_inputQ
>sequential_11_lstm_35_lstm_cell_matmul_readvariableop_resource:	ђS
@sequential_11_lstm_35_lstm_cell_matmul_1_readvariableop_resource:	 ђN
?sequential_11_lstm_35_lstm_cell_biasadd_readvariableop_resource:	ђQ
>sequential_11_lstm_36_lstm_cell_matmul_readvariableop_resource:	 ђS
@sequential_11_lstm_36_lstm_cell_matmul_1_readvariableop_resource:	 ђN
?sequential_11_lstm_36_lstm_cell_biasadd_readvariableop_resource:	ђQ
>sequential_11_lstm_37_lstm_cell_matmul_readvariableop_resource:	 ђS
@sequential_11_lstm_37_lstm_cell_matmul_1_readvariableop_resource:	 ђN
?sequential_11_lstm_37_lstm_cell_biasadd_readvariableop_resource:	ђQ
>sequential_11_lstm_38_lstm_cell_matmul_readvariableop_resource:	 ђS
@sequential_11_lstm_38_lstm_cell_matmul_1_readvariableop_resource:	 ђN
?sequential_11_lstm_38_lstm_cell_biasadd_readvariableop_resource:	ђG
5sequential_11_dense_31_matmul_readvariableop_resource:  D
6sequential_11_dense_31_biasadd_readvariableop_resource: G
5sequential_11_dense_32_matmul_readvariableop_resource: D
6sequential_11_dense_32_biasadd_readvariableop_resource:
identityѕб-sequential_11/dense_31/BiasAdd/ReadVariableOpб,sequential_11/dense_31/MatMul/ReadVariableOpб-sequential_11/dense_32/BiasAdd/ReadVariableOpб,sequential_11/dense_32/MatMul/ReadVariableOpб6sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOpб5sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOpб7sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOpбsequential_11/lstm_35/whileб6sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOpб5sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOpб7sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOpбsequential_11/lstm_36/whileб6sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOpб5sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOpб7sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOpбsequential_11/lstm_37/whileб6sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOpб5sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOpб7sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOpбsequential_11/lstm_38/whilef
sequential_11/lstm_35/ShapeShapelstm_35_input*
T0*
_output_shapes
::ь¤s
)sequential_11/lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_11/lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_11/lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_11/lstm_35/strided_sliceStridedSlice$sequential_11/lstm_35/Shape:output:02sequential_11/lstm_35/strided_slice/stack:output:04sequential_11/lstm_35/strided_slice/stack_1:output:04sequential_11/lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_11/lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : х
"sequential_11/lstm_35/zeros/packedPack,sequential_11/lstm_35/strided_slice:output:0-sequential_11/lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_11/lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_11/lstm_35/zerosFill+sequential_11/lstm_35/zeros/packed:output:0*sequential_11/lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:          h
&sequential_11/lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ╣
$sequential_11/lstm_35/zeros_1/packedPack,sequential_11/lstm_35/strided_slice:output:0/sequential_11/lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_11/lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_11/lstm_35/zeros_1Fill-sequential_11/lstm_35/zeros_1/packed:output:0,sequential_11/lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:          y
$sequential_11/lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
sequential_11/lstm_35/transpose	Transposelstm_35_input-sequential_11/lstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:
         ~
sequential_11/lstm_35/Shape_1Shape#sequential_11/lstm_35/transpose:y:0*
T0*
_output_shapes
::ь¤u
+sequential_11/lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_11/lstm_35/strided_slice_1StridedSlice&sequential_11/lstm_35/Shape_1:output:04sequential_11/lstm_35/strided_slice_1/stack:output:06sequential_11/lstm_35/strided_slice_1/stack_1:output:06sequential_11/lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_11/lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_11/lstm_35/TensorArrayV2TensorListReserve:sequential_11/lstm_35/TensorArrayV2/element_shape:output:0.sequential_11/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_11/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       б
=sequential_11/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_35/transpose:y:0Tsequential_11/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_11/lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_11/lstm_35/strided_slice_2StridedSlice#sequential_11/lstm_35/transpose:y:04sequential_11/lstm_35/strided_slice_2/stack:output:06sequential_11/lstm_35/strided_slice_2/stack_1:output:06sequential_11/lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskх
5sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOpReadVariableOp>sequential_11_lstm_35_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0м
&sequential_11/lstm_35/lstm_cell/MatMulMatMul.sequential_11/lstm_35/strided_slice_2:output:0=sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╣
7sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_11_lstm_35_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0╠
(sequential_11/lstm_35/lstm_cell/MatMul_1MatMul$sequential_11/lstm_35/zeros:output:0?sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┼
#sequential_11/lstm_35/lstm_cell/addAddV20sequential_11/lstm_35/lstm_cell/MatMul:product:02sequential_11/lstm_35/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ│
6sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?sequential_11_lstm_35_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╬
'sequential_11/lstm_35/lstm_cell/BiasAddBiasAdd'sequential_11/lstm_35/lstm_cell/add:z:0>sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
/sequential_11/lstm_35/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%sequential_11/lstm_35/lstm_cell/splitSplit8sequential_11/lstm_35/lstm_cell/split/split_dim:output:00sequential_11/lstm_35/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitћ
'sequential_11/lstm_35/lstm_cell/SigmoidSigmoid.sequential_11/lstm_35/lstm_cell/split:output:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_35/lstm_cell/Sigmoid_1Sigmoid.sequential_11/lstm_35/lstm_cell/split:output:1*
T0*'
_output_shapes
:          │
#sequential_11/lstm_35/lstm_cell/mulMul-sequential_11/lstm_35/lstm_cell/Sigmoid_1:y:0&sequential_11/lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:          ј
$sequential_11/lstm_35/lstm_cell/ReluRelu.sequential_11/lstm_35/lstm_cell/split:output:2*
T0*'
_output_shapes
:          ┐
%sequential_11/lstm_35/lstm_cell/mul_1Mul+sequential_11/lstm_35/lstm_cell/Sigmoid:y:02sequential_11/lstm_35/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ┤
%sequential_11/lstm_35/lstm_cell/add_1AddV2'sequential_11/lstm_35/lstm_cell/mul:z:0)sequential_11/lstm_35/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_35/lstm_cell/Sigmoid_2Sigmoid.sequential_11/lstm_35/lstm_cell/split:output:3*
T0*'
_output_shapes
:          І
&sequential_11/lstm_35/lstm_cell/Relu_1Relu)sequential_11/lstm_35/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          ├
%sequential_11/lstm_35/lstm_cell/mul_2Mul-sequential_11/lstm_35/lstm_cell/Sigmoid_2:y:04sequential_11/lstm_35/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ё
3sequential_11/lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Щ
%sequential_11/lstm_35/TensorArrayV2_1TensorListReserve<sequential_11/lstm_35/TensorArrayV2_1/element_shape:output:0.sequential_11/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_11/lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_11/lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_11/lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
sequential_11/lstm_35/whileWhile1sequential_11/lstm_35/while/loop_counter:output:07sequential_11/lstm_35/while/maximum_iterations:output:0#sequential_11/lstm_35/time:output:0.sequential_11/lstm_35/TensorArrayV2_1:handle:0$sequential_11/lstm_35/zeros:output:0&sequential_11/lstm_35/zeros_1:output:0.sequential_11/lstm_35/strided_slice_1:output:0Msequential_11/lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_11_lstm_35_lstm_cell_matmul_readvariableop_resource@sequential_11_lstm_35_lstm_cell_matmul_1_readvariableop_resource?sequential_11_lstm_35_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_11_lstm_35_while_body_759203*3
cond+R)
'sequential_11_lstm_35_while_cond_759202*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ќ
Fsequential_11/lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ё
8sequential_11/lstm_35/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_35/while:output:3Osequential_11/lstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0~
+sequential_11/lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_11/lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
%sequential_11/lstm_35/strided_slice_3StridedSliceAsequential_11/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_35/strided_slice_3/stack:output:06sequential_11/lstm_35/strided_slice_3/stack_1:output:06sequential_11/lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask{
&sequential_11/lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
!sequential_11/lstm_35/transpose_1	TransposeAsequential_11/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 q
sequential_11/lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    і
!sequential_11/dropout_55/IdentityIdentity%sequential_11/lstm_35/transpose_1:y:0*
T0*+
_output_shapes
:         
 Ѓ
sequential_11/lstm_36/ShapeShape*sequential_11/dropout_55/Identity:output:0*
T0*
_output_shapes
::ь¤s
)sequential_11/lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_11/lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_11/lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_11/lstm_36/strided_sliceStridedSlice$sequential_11/lstm_36/Shape:output:02sequential_11/lstm_36/strided_slice/stack:output:04sequential_11/lstm_36/strided_slice/stack_1:output:04sequential_11/lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_11/lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : х
"sequential_11/lstm_36/zeros/packedPack,sequential_11/lstm_36/strided_slice:output:0-sequential_11/lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_11/lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_11/lstm_36/zerosFill+sequential_11/lstm_36/zeros/packed:output:0*sequential_11/lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:          h
&sequential_11/lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ╣
$sequential_11/lstm_36/zeros_1/packedPack,sequential_11/lstm_36/strided_slice:output:0/sequential_11/lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_11/lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_11/lstm_36/zeros_1Fill-sequential_11/lstm_36/zeros_1/packed:output:0,sequential_11/lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:          y
$sequential_11/lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
sequential_11/lstm_36/transpose	Transpose*sequential_11/dropout_55/Identity:output:0-sequential_11/lstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:
          ~
sequential_11/lstm_36/Shape_1Shape#sequential_11/lstm_36/transpose:y:0*
T0*
_output_shapes
::ь¤u
+sequential_11/lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_11/lstm_36/strided_slice_1StridedSlice&sequential_11/lstm_36/Shape_1:output:04sequential_11/lstm_36/strided_slice_1/stack:output:06sequential_11/lstm_36/strided_slice_1/stack_1:output:06sequential_11/lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_11/lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_11/lstm_36/TensorArrayV2TensorListReserve:sequential_11/lstm_36/TensorArrayV2/element_shape:output:0.sequential_11/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_11/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        б
=sequential_11/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_36/transpose:y:0Tsequential_11/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_11/lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_11/lstm_36/strided_slice_2StridedSlice#sequential_11/lstm_36/transpose:y:04sequential_11/lstm_36/strided_slice_2/stack:output:06sequential_11/lstm_36/strided_slice_2/stack_1:output:06sequential_11/lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskх
5sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOpReadVariableOp>sequential_11_lstm_36_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0м
&sequential_11/lstm_36/lstm_cell/MatMulMatMul.sequential_11/lstm_36/strided_slice_2:output:0=sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╣
7sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_11_lstm_36_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0╠
(sequential_11/lstm_36/lstm_cell/MatMul_1MatMul$sequential_11/lstm_36/zeros:output:0?sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┼
#sequential_11/lstm_36/lstm_cell/addAddV20sequential_11/lstm_36/lstm_cell/MatMul:product:02sequential_11/lstm_36/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ│
6sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?sequential_11_lstm_36_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╬
'sequential_11/lstm_36/lstm_cell/BiasAddBiasAdd'sequential_11/lstm_36/lstm_cell/add:z:0>sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
/sequential_11/lstm_36/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%sequential_11/lstm_36/lstm_cell/splitSplit8sequential_11/lstm_36/lstm_cell/split/split_dim:output:00sequential_11/lstm_36/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitћ
'sequential_11/lstm_36/lstm_cell/SigmoidSigmoid.sequential_11/lstm_36/lstm_cell/split:output:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_36/lstm_cell/Sigmoid_1Sigmoid.sequential_11/lstm_36/lstm_cell/split:output:1*
T0*'
_output_shapes
:          │
#sequential_11/lstm_36/lstm_cell/mulMul-sequential_11/lstm_36/lstm_cell/Sigmoid_1:y:0&sequential_11/lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:          ј
$sequential_11/lstm_36/lstm_cell/ReluRelu.sequential_11/lstm_36/lstm_cell/split:output:2*
T0*'
_output_shapes
:          ┐
%sequential_11/lstm_36/lstm_cell/mul_1Mul+sequential_11/lstm_36/lstm_cell/Sigmoid:y:02sequential_11/lstm_36/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ┤
%sequential_11/lstm_36/lstm_cell/add_1AddV2'sequential_11/lstm_36/lstm_cell/mul:z:0)sequential_11/lstm_36/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_36/lstm_cell/Sigmoid_2Sigmoid.sequential_11/lstm_36/lstm_cell/split:output:3*
T0*'
_output_shapes
:          І
&sequential_11/lstm_36/lstm_cell/Relu_1Relu)sequential_11/lstm_36/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          ├
%sequential_11/lstm_36/lstm_cell/mul_2Mul-sequential_11/lstm_36/lstm_cell/Sigmoid_2:y:04sequential_11/lstm_36/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ё
3sequential_11/lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Щ
%sequential_11/lstm_36/TensorArrayV2_1TensorListReserve<sequential_11/lstm_36/TensorArrayV2_1/element_shape:output:0.sequential_11/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_11/lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_11/lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_11/lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
sequential_11/lstm_36/whileWhile1sequential_11/lstm_36/while/loop_counter:output:07sequential_11/lstm_36/while/maximum_iterations:output:0#sequential_11/lstm_36/time:output:0.sequential_11/lstm_36/TensorArrayV2_1:handle:0$sequential_11/lstm_36/zeros:output:0&sequential_11/lstm_36/zeros_1:output:0.sequential_11/lstm_36/strided_slice_1:output:0Msequential_11/lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_11_lstm_36_lstm_cell_matmul_readvariableop_resource@sequential_11_lstm_36_lstm_cell_matmul_1_readvariableop_resource?sequential_11_lstm_36_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_11_lstm_36_while_body_759343*3
cond+R)
'sequential_11_lstm_36_while_cond_759342*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ќ
Fsequential_11/lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ё
8sequential_11/lstm_36/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_36/while:output:3Osequential_11/lstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0~
+sequential_11/lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_11/lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
%sequential_11/lstm_36/strided_slice_3StridedSliceAsequential_11/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_36/strided_slice_3/stack:output:06sequential_11/lstm_36/strided_slice_3/stack_1:output:06sequential_11/lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask{
&sequential_11/lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
!sequential_11/lstm_36/transpose_1	TransposeAsequential_11/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 q
sequential_11/lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    і
!sequential_11/dropout_56/IdentityIdentity%sequential_11/lstm_36/transpose_1:y:0*
T0*+
_output_shapes
:         
 Ѓ
sequential_11/lstm_37/ShapeShape*sequential_11/dropout_56/Identity:output:0*
T0*
_output_shapes
::ь¤s
)sequential_11/lstm_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_11/lstm_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_11/lstm_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_11/lstm_37/strided_sliceStridedSlice$sequential_11/lstm_37/Shape:output:02sequential_11/lstm_37/strided_slice/stack:output:04sequential_11/lstm_37/strided_slice/stack_1:output:04sequential_11/lstm_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_11/lstm_37/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : х
"sequential_11/lstm_37/zeros/packedPack,sequential_11/lstm_37/strided_slice:output:0-sequential_11/lstm_37/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_11/lstm_37/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_11/lstm_37/zerosFill+sequential_11/lstm_37/zeros/packed:output:0*sequential_11/lstm_37/zeros/Const:output:0*
T0*'
_output_shapes
:          h
&sequential_11/lstm_37/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ╣
$sequential_11/lstm_37/zeros_1/packedPack,sequential_11/lstm_37/strided_slice:output:0/sequential_11/lstm_37/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_11/lstm_37/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_11/lstm_37/zeros_1Fill-sequential_11/lstm_37/zeros_1/packed:output:0,sequential_11/lstm_37/zeros_1/Const:output:0*
T0*'
_output_shapes
:          y
$sequential_11/lstm_37/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
sequential_11/lstm_37/transpose	Transpose*sequential_11/dropout_56/Identity:output:0-sequential_11/lstm_37/transpose/perm:output:0*
T0*+
_output_shapes
:
          ~
sequential_11/lstm_37/Shape_1Shape#sequential_11/lstm_37/transpose:y:0*
T0*
_output_shapes
::ь¤u
+sequential_11/lstm_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_11/lstm_37/strided_slice_1StridedSlice&sequential_11/lstm_37/Shape_1:output:04sequential_11/lstm_37/strided_slice_1/stack:output:06sequential_11/lstm_37/strided_slice_1/stack_1:output:06sequential_11/lstm_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_11/lstm_37/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_11/lstm_37/TensorArrayV2TensorListReserve:sequential_11/lstm_37/TensorArrayV2/element_shape:output:0.sequential_11/lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_11/lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        б
=sequential_11/lstm_37/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_37/transpose:y:0Tsequential_11/lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_11/lstm_37/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_37/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_37/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_11/lstm_37/strided_slice_2StridedSlice#sequential_11/lstm_37/transpose:y:04sequential_11/lstm_37/strided_slice_2/stack:output:06sequential_11/lstm_37/strided_slice_2/stack_1:output:06sequential_11/lstm_37/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskх
5sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOpReadVariableOp>sequential_11_lstm_37_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0м
&sequential_11/lstm_37/lstm_cell/MatMulMatMul.sequential_11/lstm_37/strided_slice_2:output:0=sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╣
7sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_11_lstm_37_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0╠
(sequential_11/lstm_37/lstm_cell/MatMul_1MatMul$sequential_11/lstm_37/zeros:output:0?sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┼
#sequential_11/lstm_37/lstm_cell/addAddV20sequential_11/lstm_37/lstm_cell/MatMul:product:02sequential_11/lstm_37/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ│
6sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?sequential_11_lstm_37_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╬
'sequential_11/lstm_37/lstm_cell/BiasAddBiasAdd'sequential_11/lstm_37/lstm_cell/add:z:0>sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
/sequential_11/lstm_37/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%sequential_11/lstm_37/lstm_cell/splitSplit8sequential_11/lstm_37/lstm_cell/split/split_dim:output:00sequential_11/lstm_37/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitћ
'sequential_11/lstm_37/lstm_cell/SigmoidSigmoid.sequential_11/lstm_37/lstm_cell/split:output:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_37/lstm_cell/Sigmoid_1Sigmoid.sequential_11/lstm_37/lstm_cell/split:output:1*
T0*'
_output_shapes
:          │
#sequential_11/lstm_37/lstm_cell/mulMul-sequential_11/lstm_37/lstm_cell/Sigmoid_1:y:0&sequential_11/lstm_37/zeros_1:output:0*
T0*'
_output_shapes
:          ј
$sequential_11/lstm_37/lstm_cell/ReluRelu.sequential_11/lstm_37/lstm_cell/split:output:2*
T0*'
_output_shapes
:          ┐
%sequential_11/lstm_37/lstm_cell/mul_1Mul+sequential_11/lstm_37/lstm_cell/Sigmoid:y:02sequential_11/lstm_37/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ┤
%sequential_11/lstm_37/lstm_cell/add_1AddV2'sequential_11/lstm_37/lstm_cell/mul:z:0)sequential_11/lstm_37/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_37/lstm_cell/Sigmoid_2Sigmoid.sequential_11/lstm_37/lstm_cell/split:output:3*
T0*'
_output_shapes
:          І
&sequential_11/lstm_37/lstm_cell/Relu_1Relu)sequential_11/lstm_37/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          ├
%sequential_11/lstm_37/lstm_cell/mul_2Mul-sequential_11/lstm_37/lstm_cell/Sigmoid_2:y:04sequential_11/lstm_37/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ё
3sequential_11/lstm_37/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Щ
%sequential_11/lstm_37/TensorArrayV2_1TensorListReserve<sequential_11/lstm_37/TensorArrayV2_1/element_shape:output:0.sequential_11/lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_11/lstm_37/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_11/lstm_37/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_11/lstm_37/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
sequential_11/lstm_37/whileWhile1sequential_11/lstm_37/while/loop_counter:output:07sequential_11/lstm_37/while/maximum_iterations:output:0#sequential_11/lstm_37/time:output:0.sequential_11/lstm_37/TensorArrayV2_1:handle:0$sequential_11/lstm_37/zeros:output:0&sequential_11/lstm_37/zeros_1:output:0.sequential_11/lstm_37/strided_slice_1:output:0Msequential_11/lstm_37/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_11_lstm_37_lstm_cell_matmul_readvariableop_resource@sequential_11_lstm_37_lstm_cell_matmul_1_readvariableop_resource?sequential_11_lstm_37_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_11_lstm_37_while_body_759483*3
cond+R)
'sequential_11_lstm_37_while_cond_759482*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ќ
Fsequential_11/lstm_37/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ё
8sequential_11/lstm_37/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_37/while:output:3Osequential_11/lstm_37/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0~
+sequential_11/lstm_37/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_11/lstm_37/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_37/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
%sequential_11/lstm_37/strided_slice_3StridedSliceAsequential_11/lstm_37/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_37/strided_slice_3/stack:output:06sequential_11/lstm_37/strided_slice_3/stack_1:output:06sequential_11/lstm_37/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask{
&sequential_11/lstm_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
!sequential_11/lstm_37/transpose_1	TransposeAsequential_11/lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_37/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 q
sequential_11/lstm_37/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    і
!sequential_11/dropout_57/IdentityIdentity%sequential_11/lstm_37/transpose_1:y:0*
T0*+
_output_shapes
:         
 Ѓ
sequential_11/lstm_38/ShapeShape*sequential_11/dropout_57/Identity:output:0*
T0*
_output_shapes
::ь¤s
)sequential_11/lstm_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_11/lstm_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_11/lstm_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_11/lstm_38/strided_sliceStridedSlice$sequential_11/lstm_38/Shape:output:02sequential_11/lstm_38/strided_slice/stack:output:04sequential_11/lstm_38/strided_slice/stack_1:output:04sequential_11/lstm_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_11/lstm_38/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : х
"sequential_11/lstm_38/zeros/packedPack,sequential_11/lstm_38/strided_slice:output:0-sequential_11/lstm_38/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_11/lstm_38/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_11/lstm_38/zerosFill+sequential_11/lstm_38/zeros/packed:output:0*sequential_11/lstm_38/zeros/Const:output:0*
T0*'
_output_shapes
:          h
&sequential_11/lstm_38/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ╣
$sequential_11/lstm_38/zeros_1/packedPack,sequential_11/lstm_38/strided_slice:output:0/sequential_11/lstm_38/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_11/lstm_38/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_11/lstm_38/zeros_1Fill-sequential_11/lstm_38/zeros_1/packed:output:0,sequential_11/lstm_38/zeros_1/Const:output:0*
T0*'
_output_shapes
:          y
$sequential_11/lstm_38/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
sequential_11/lstm_38/transpose	Transpose*sequential_11/dropout_57/Identity:output:0-sequential_11/lstm_38/transpose/perm:output:0*
T0*+
_output_shapes
:
          ~
sequential_11/lstm_38/Shape_1Shape#sequential_11/lstm_38/transpose:y:0*
T0*
_output_shapes
::ь¤u
+sequential_11/lstm_38/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_38/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_38/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_11/lstm_38/strided_slice_1StridedSlice&sequential_11/lstm_38/Shape_1:output:04sequential_11/lstm_38/strided_slice_1/stack:output:06sequential_11/lstm_38/strided_slice_1/stack_1:output:06sequential_11/lstm_38/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_11/lstm_38/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_11/lstm_38/TensorArrayV2TensorListReserve:sequential_11/lstm_38/TensorArrayV2/element_shape:output:0.sequential_11/lstm_38/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_11/lstm_38/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        б
=sequential_11/lstm_38/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_38/transpose:y:0Tsequential_11/lstm_38/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_11/lstm_38/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_38/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_11/lstm_38/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_11/lstm_38/strided_slice_2StridedSlice#sequential_11/lstm_38/transpose:y:04sequential_11/lstm_38/strided_slice_2/stack:output:06sequential_11/lstm_38/strided_slice_2/stack_1:output:06sequential_11/lstm_38/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskх
5sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOpReadVariableOp>sequential_11_lstm_38_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0м
&sequential_11/lstm_38/lstm_cell/MatMulMatMul.sequential_11/lstm_38/strided_slice_2:output:0=sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╣
7sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_11_lstm_38_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0╠
(sequential_11/lstm_38/lstm_cell/MatMul_1MatMul$sequential_11/lstm_38/zeros:output:0?sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┼
#sequential_11/lstm_38/lstm_cell/addAddV20sequential_11/lstm_38/lstm_cell/MatMul:product:02sequential_11/lstm_38/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ│
6sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?sequential_11_lstm_38_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╬
'sequential_11/lstm_38/lstm_cell/BiasAddBiasAdd'sequential_11/lstm_38/lstm_cell/add:z:0>sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
/sequential_11/lstm_38/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%sequential_11/lstm_38/lstm_cell/splitSplit8sequential_11/lstm_38/lstm_cell/split/split_dim:output:00sequential_11/lstm_38/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitћ
'sequential_11/lstm_38/lstm_cell/SigmoidSigmoid.sequential_11/lstm_38/lstm_cell/split:output:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_38/lstm_cell/Sigmoid_1Sigmoid.sequential_11/lstm_38/lstm_cell/split:output:1*
T0*'
_output_shapes
:          │
#sequential_11/lstm_38/lstm_cell/mulMul-sequential_11/lstm_38/lstm_cell/Sigmoid_1:y:0&sequential_11/lstm_38/zeros_1:output:0*
T0*'
_output_shapes
:          ј
$sequential_11/lstm_38/lstm_cell/ReluRelu.sequential_11/lstm_38/lstm_cell/split:output:2*
T0*'
_output_shapes
:          ┐
%sequential_11/lstm_38/lstm_cell/mul_1Mul+sequential_11/lstm_38/lstm_cell/Sigmoid:y:02sequential_11/lstm_38/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ┤
%sequential_11/lstm_38/lstm_cell/add_1AddV2'sequential_11/lstm_38/lstm_cell/mul:z:0)sequential_11/lstm_38/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          ќ
)sequential_11/lstm_38/lstm_cell/Sigmoid_2Sigmoid.sequential_11/lstm_38/lstm_cell/split:output:3*
T0*'
_output_shapes
:          І
&sequential_11/lstm_38/lstm_cell/Relu_1Relu)sequential_11/lstm_38/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          ├
%sequential_11/lstm_38/lstm_cell/mul_2Mul-sequential_11/lstm_38/lstm_cell/Sigmoid_2:y:04sequential_11/lstm_38/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ё
3sequential_11/lstm_38/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        t
2sequential_11/lstm_38/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Є
%sequential_11/lstm_38/TensorArrayV2_1TensorListReserve<sequential_11/lstm_38/TensorArrayV2_1/element_shape:output:0;sequential_11/lstm_38/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_11/lstm_38/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_11/lstm_38/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_11/lstm_38/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
sequential_11/lstm_38/whileWhile1sequential_11/lstm_38/while/loop_counter:output:07sequential_11/lstm_38/while/maximum_iterations:output:0#sequential_11/lstm_38/time:output:0.sequential_11/lstm_38/TensorArrayV2_1:handle:0$sequential_11/lstm_38/zeros:output:0&sequential_11/lstm_38/zeros_1:output:0.sequential_11/lstm_38/strided_slice_1:output:0Msequential_11/lstm_38/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_11_lstm_38_lstm_cell_matmul_readvariableop_resource@sequential_11_lstm_38_lstm_cell_matmul_1_readvariableop_resource?sequential_11_lstm_38_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_11_lstm_38_while_body_759624*3
cond+R)
'sequential_11_lstm_38_while_cond_759623*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ќ
Fsequential_11/lstm_38/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ў
8sequential_11/lstm_38/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_38/while:output:3Osequential_11/lstm_38/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elements~
+sequential_11/lstm_38/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_11/lstm_38/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_11/lstm_38/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
%sequential_11/lstm_38/strided_slice_3StridedSliceAsequential_11/lstm_38/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_38/strided_slice_3/stack:output:06sequential_11/lstm_38/strided_slice_3/stack_1:output:06sequential_11/lstm_38/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask{
&sequential_11/lstm_38/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
!sequential_11/lstm_38/transpose_1	TransposeAsequential_11/lstm_38/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_38/transpose_1/perm:output:0*
T0*+
_output_shapes
:          q
sequential_11/lstm_38/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ј
!sequential_11/dropout_58/IdentityIdentity.sequential_11/lstm_38/strided_slice_3:output:0*
T0*'
_output_shapes
:          б
,sequential_11/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_31_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0╗
sequential_11/dense_31/MatMulMatMul*sequential_11/dropout_58/Identity:output:04sequential_11/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          а
-sequential_11/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
sequential_11/dense_31/BiasAddBiasAdd'sequential_11/dense_31/MatMul:product:05sequential_11/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ~
sequential_11/dense_31/ReluRelu'sequential_11/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:          і
!sequential_11/dropout_59/IdentityIdentity)sequential_11/dense_31/Relu:activations:0*
T0*'
_output_shapes
:          б
,sequential_11/dense_32/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_32_matmul_readvariableop_resource*
_output_shapes

: *
dtype0╗
sequential_11/dense_32/MatMulMatMul*sequential_11/dropout_59/Identity:output:04sequential_11/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_11/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_11/dense_32/BiasAddBiasAdd'sequential_11/dense_32/MatMul:product:05sequential_11/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_11/dense_32/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ё
NoOpNoOp.^sequential_11/dense_31/BiasAdd/ReadVariableOp-^sequential_11/dense_31/MatMul/ReadVariableOp.^sequential_11/dense_32/BiasAdd/ReadVariableOp-^sequential_11/dense_32/MatMul/ReadVariableOp7^sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOp6^sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOp8^sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOp^sequential_11/lstm_35/while7^sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOp6^sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOp8^sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOp^sequential_11/lstm_36/while7^sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOp6^sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOp8^sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOp^sequential_11/lstm_37/while7^sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOp6^sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOp8^sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOp^sequential_11/lstm_38/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 2^
-sequential_11/dense_31/BiasAdd/ReadVariableOp-sequential_11/dense_31/BiasAdd/ReadVariableOp2\
,sequential_11/dense_31/MatMul/ReadVariableOp,sequential_11/dense_31/MatMul/ReadVariableOp2^
-sequential_11/dense_32/BiasAdd/ReadVariableOp-sequential_11/dense_32/BiasAdd/ReadVariableOp2\
,sequential_11/dense_32/MatMul/ReadVariableOp,sequential_11/dense_32/MatMul/ReadVariableOp2p
6sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOp6sequential_11/lstm_35/lstm_cell/BiasAdd/ReadVariableOp2n
5sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOp5sequential_11/lstm_35/lstm_cell/MatMul/ReadVariableOp2r
7sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOp7sequential_11/lstm_35/lstm_cell/MatMul_1/ReadVariableOp2:
sequential_11/lstm_35/whilesequential_11/lstm_35/while2p
6sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOp6sequential_11/lstm_36/lstm_cell/BiasAdd/ReadVariableOp2n
5sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOp5sequential_11/lstm_36/lstm_cell/MatMul/ReadVariableOp2r
7sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOp7sequential_11/lstm_36/lstm_cell/MatMul_1/ReadVariableOp2:
sequential_11/lstm_36/whilesequential_11/lstm_36/while2p
6sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOp6sequential_11/lstm_37/lstm_cell/BiasAdd/ReadVariableOp2n
5sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOp5sequential_11/lstm_37/lstm_cell/MatMul/ReadVariableOp2r
7sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOp7sequential_11/lstm_37/lstm_cell/MatMul_1/ReadVariableOp2:
sequential_11/lstm_37/whilesequential_11/lstm_37/while2p
6sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOp6sequential_11/lstm_38/lstm_cell/BiasAdd/ReadVariableOp2n
5sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOp5sequential_11/lstm_38/lstm_cell/MatMul/ReadVariableOp2r
7sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOp7sequential_11/lstm_38/lstm_cell/MatMul_1/ReadVariableOp2:
sequential_11/lstm_38/whilesequential_11/lstm_38/while:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
┘
d
F__inference_dropout_59_layer_call_and_return_conditional_losses_762452

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ж8
х
while_body_764262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765768

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
с
и
(__inference_lstm_38_layer_call_fn_764670
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_760909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764666:&"
 
_user_specified_name764664:&"
 
_user_specified_name764662:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
╩I
Ѕ
C__inference_lstm_35_layer_call_and_return_conditional_losses_761959

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_761875*
condR
while_cond_761874*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ж
d
F__inference_dropout_57_layer_call_and_return_conditional_losses_764659

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╣
з
*__inference_lstm_cell_layer_call_fn_765704

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765696:&"
 
_user_specified_name765694:&"
 
_user_specified_name765692:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
є:
х
while_body_764908
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ж8
х
while_body_762031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И8
щ
C__inference_lstm_37_layer_call_and_return_conditional_losses_760561

inputs#
lstm_cell_760479:	 ђ#
lstm_cell_760481:	 ђ
lstm_cell_760483:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760479lstm_cell_760481lstm_cell_760483*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760478n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760479lstm_cell_760481lstm_cell_760483*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760492*
condR
while_cond_760491*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760483:&"
 
_user_specified_name760481:&"
 
_user_specified_name760479:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
┬	
├
while_cond_763118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763118___redundant_placeholder04
0while_while_cond_763118___redundant_placeholder14
0while_while_cond_763118___redundant_placeholder24
0while_while_cond_763118___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_764404
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764404___redundant_placeholder04
0while_while_cond_764404___redundant_placeholder14
0while_while_cond_764404___redundant_placeholder24
0while_while_cond_764404___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Х

e
F__inference_dropout_57_layer_call_and_return_conditional_losses_761602

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765670

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ИS
э
'sequential_11_lstm_35_while_body_759203H
Dsequential_11_lstm_35_while_sequential_11_lstm_35_while_loop_counterN
Jsequential_11_lstm_35_while_sequential_11_lstm_35_while_maximum_iterations+
'sequential_11_lstm_35_while_placeholder-
)sequential_11_lstm_35_while_placeholder_1-
)sequential_11_lstm_35_while_placeholder_2-
)sequential_11_lstm_35_while_placeholder_3G
Csequential_11_lstm_35_while_sequential_11_lstm_35_strided_slice_1_0Ѓ
sequential_11_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_35_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_11_lstm_35_while_lstm_cell_matmul_readvariableop_resource_0:	ђ[
Hsequential_11_lstm_35_while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђV
Gsequential_11_lstm_35_while_lstm_cell_biasadd_readvariableop_resource_0:	ђ(
$sequential_11_lstm_35_while_identity*
&sequential_11_lstm_35_while_identity_1*
&sequential_11_lstm_35_while_identity_2*
&sequential_11_lstm_35_while_identity_3*
&sequential_11_lstm_35_while_identity_4*
&sequential_11_lstm_35_while_identity_5E
Asequential_11_lstm_35_while_sequential_11_lstm_35_strided_slice_1Ђ
}sequential_11_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_35_tensorarrayunstack_tensorlistfromtensorW
Dsequential_11_lstm_35_while_lstm_cell_matmul_readvariableop_resource:	ђY
Fsequential_11_lstm_35_while_lstm_cell_matmul_1_readvariableop_resource:	 ђT
Esequential_11_lstm_35_while_lstm_cell_biasadd_readvariableop_resource:	ђѕб<sequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOpб;sequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOpб=sequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOpъ
Msequential_11/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?sequential_11/lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_35_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_35_while_placeholderVsequential_11/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0├
;sequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpFsequential_11_lstm_35_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0Ш
,sequential_11/lstm_35/while/lstm_cell/MatMulMatMulFsequential_11/lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђК
=sequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHsequential_11_lstm_35_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0П
.sequential_11/lstm_35/while/lstm_cell/MatMul_1MatMul)sequential_11_lstm_35_while_placeholder_2Esequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђО
)sequential_11/lstm_35/while/lstm_cell/addAddV26sequential_11/lstm_35/while/lstm_cell/MatMul:product:08sequential_11/lstm_35/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ┴
<sequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGsequential_11_lstm_35_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0Я
-sequential_11/lstm_35/while/lstm_cell/BiasAddBiasAdd-sequential_11/lstm_35/while/lstm_cell/add:z:0Dsequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
5sequential_11/lstm_35/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :е
+sequential_11/lstm_35/while/lstm_cell/splitSplit>sequential_11/lstm_35/while/lstm_cell/split/split_dim:output:06sequential_11/lstm_35/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitа
-sequential_11/lstm_35/while/lstm_cell/SigmoidSigmoid4sequential_11/lstm_35/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_35/while/lstm_cell/Sigmoid_1Sigmoid4sequential_11/lstm_35/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ┬
)sequential_11/lstm_35/while/lstm_cell/mulMul3sequential_11/lstm_35/while/lstm_cell/Sigmoid_1:y:0)sequential_11_lstm_35_while_placeholder_3*
T0*'
_output_shapes
:          џ
*sequential_11/lstm_35/while/lstm_cell/ReluRelu4sequential_11/lstm_35/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Л
+sequential_11/lstm_35/while/lstm_cell/mul_1Mul1sequential_11/lstm_35/while/lstm_cell/Sigmoid:y:08sequential_11/lstm_35/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          к
+sequential_11/lstm_35/while/lstm_cell/add_1AddV2-sequential_11/lstm_35/while/lstm_cell/mul:z:0/sequential_11/lstm_35/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_35/while/lstm_cell/Sigmoid_2Sigmoid4sequential_11/lstm_35/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:          Ќ
,sequential_11/lstm_35/while/lstm_cell/Relu_1Relu/sequential_11/lstm_35/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Н
+sequential_11/lstm_35/while/lstm_cell/mul_2Mul3sequential_11/lstm_35/while/lstm_cell/Sigmoid_2:y:0:sequential_11/lstm_35/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          џ
@sequential_11/lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_35_while_placeholder_1'sequential_11_lstm_35_while_placeholder/sequential_11/lstm_35/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_11/lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_11/lstm_35/while/addAddV2'sequential_11_lstm_35_while_placeholder*sequential_11/lstm_35/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_11/lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_11/lstm_35/while/add_1AddV2Dsequential_11_lstm_35_while_sequential_11_lstm_35_while_loop_counter,sequential_11/lstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_11/lstm_35/while/IdentityIdentity%sequential_11/lstm_35/while/add_1:z:0!^sequential_11/lstm_35/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_11/lstm_35/while/Identity_1IdentityJsequential_11_lstm_35_while_sequential_11_lstm_35_while_maximum_iterations!^sequential_11/lstm_35/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_11/lstm_35/while/Identity_2Identity#sequential_11/lstm_35/while/add:z:0!^sequential_11/lstm_35/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_11/lstm_35/while/Identity_3IdentityPsequential_11/lstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_35/while/NoOp*
T0*
_output_shapes
: И
&sequential_11/lstm_35/while/Identity_4Identity/sequential_11/lstm_35/while/lstm_cell/mul_2:z:0!^sequential_11/lstm_35/while/NoOp*
T0*'
_output_shapes
:          И
&sequential_11/lstm_35/while/Identity_5Identity/sequential_11/lstm_35/while/lstm_cell/add_1:z:0!^sequential_11/lstm_35/while/NoOp*
T0*'
_output_shapes
:          ч
 sequential_11/lstm_35/while/NoOpNoOp=^sequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOp<^sequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOp>^sequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "Y
&sequential_11_lstm_35_while_identity_1/sequential_11/lstm_35/while/Identity_1:output:0"Y
&sequential_11_lstm_35_while_identity_2/sequential_11/lstm_35/while/Identity_2:output:0"Y
&sequential_11_lstm_35_while_identity_3/sequential_11/lstm_35/while/Identity_3:output:0"Y
&sequential_11_lstm_35_while_identity_4/sequential_11/lstm_35/while/Identity_4:output:0"Y
&sequential_11_lstm_35_while_identity_5/sequential_11/lstm_35/while/Identity_5:output:0"U
$sequential_11_lstm_35_while_identity-sequential_11/lstm_35/while/Identity:output:0"љ
Esequential_11_lstm_35_while_lstm_cell_biasadd_readvariableop_resourceGsequential_11_lstm_35_while_lstm_cell_biasadd_readvariableop_resource_0"њ
Fsequential_11_lstm_35_while_lstm_cell_matmul_1_readvariableop_resourceHsequential_11_lstm_35_while_lstm_cell_matmul_1_readvariableop_resource_0"ј
Dsequential_11_lstm_35_while_lstm_cell_matmul_readvariableop_resourceFsequential_11_lstm_35_while_lstm_cell_matmul_readvariableop_resource_0"ѕ
Asequential_11_lstm_35_while_sequential_11_lstm_35_strided_slice_1Csequential_11_lstm_35_while_sequential_11_lstm_35_strided_slice_1_0"ђ
}sequential_11_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_35_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2|
<sequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOp<sequential_11/lstm_35/while/lstm_cell/BiasAdd/ReadVariableOp2z
;sequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOp;sequential_11/lstm_35/while/lstm_cell/MatMul/ReadVariableOp2~
=sequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOp=sequential_11/lstm_35/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:uq

_output_shapes
: 
W
_user_specified_name?=sequential_11/lstm_35/TensorArrayUnstack/TensorListFromTensor:]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_35/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_35/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_35/while/loop_counter
┬$
╬
while_body_760291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_760315_0:	 ђ+
while_lstm_cell_760317_0:	 ђ'
while_lstm_cell_760319_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_760315:	 ђ)
while_lstm_cell_760317:	 ђ%
while_lstm_cell_760319:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_760315_0while_lstm_cell_760317_0while_lstm_cell_760319_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760277┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_760315while_lstm_cell_760315_0"2
while_lstm_cell_760317while_lstm_cell_760317_0"2
while_lstm_cell_760319while_lstm_cell_760319_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name760319:&	"
 
_user_specified_name760317:&"
 
_user_specified_name760315:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
џ

e
F__inference_dropout_58_layer_call_and_return_conditional_losses_765305

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:          a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
вT
э
'sequential_11_lstm_38_while_body_759624H
Dsequential_11_lstm_38_while_sequential_11_lstm_38_while_loop_counterN
Jsequential_11_lstm_38_while_sequential_11_lstm_38_while_maximum_iterations+
'sequential_11_lstm_38_while_placeholder-
)sequential_11_lstm_38_while_placeholder_1-
)sequential_11_lstm_38_while_placeholder_2-
)sequential_11_lstm_38_while_placeholder_3G
Csequential_11_lstm_38_while_sequential_11_lstm_38_strided_slice_1_0Ѓ
sequential_11_lstm_38_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_38_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_11_lstm_38_while_lstm_cell_matmul_readvariableop_resource_0:	 ђ[
Hsequential_11_lstm_38_while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђV
Gsequential_11_lstm_38_while_lstm_cell_biasadd_readvariableop_resource_0:	ђ(
$sequential_11_lstm_38_while_identity*
&sequential_11_lstm_38_while_identity_1*
&sequential_11_lstm_38_while_identity_2*
&sequential_11_lstm_38_while_identity_3*
&sequential_11_lstm_38_while_identity_4*
&sequential_11_lstm_38_while_identity_5E
Asequential_11_lstm_38_while_sequential_11_lstm_38_strided_slice_1Ђ
}sequential_11_lstm_38_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_38_tensorarrayunstack_tensorlistfromtensorW
Dsequential_11_lstm_38_while_lstm_cell_matmul_readvariableop_resource:	 ђY
Fsequential_11_lstm_38_while_lstm_cell_matmul_1_readvariableop_resource:	 ђT
Esequential_11_lstm_38_while_lstm_cell_biasadd_readvariableop_resource:	ђѕб<sequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOpб;sequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOpб=sequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOpъ
Msequential_11/lstm_38/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ћ
?sequential_11/lstm_38/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_38_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_38_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_38_while_placeholderVsequential_11/lstm_38/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0├
;sequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpFsequential_11_lstm_38_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Ш
,sequential_11/lstm_38/while/lstm_cell/MatMulMatMulFsequential_11/lstm_38/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђК
=sequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHsequential_11_lstm_38_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0П
.sequential_11/lstm_38/while/lstm_cell/MatMul_1MatMul)sequential_11_lstm_38_while_placeholder_2Esequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђО
)sequential_11/lstm_38/while/lstm_cell/addAddV26sequential_11/lstm_38/while/lstm_cell/MatMul:product:08sequential_11/lstm_38/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ┴
<sequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGsequential_11_lstm_38_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0Я
-sequential_11/lstm_38/while/lstm_cell/BiasAddBiasAdd-sequential_11/lstm_38/while/lstm_cell/add:z:0Dsequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
5sequential_11/lstm_38/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :е
+sequential_11/lstm_38/while/lstm_cell/splitSplit>sequential_11/lstm_38/while/lstm_cell/split/split_dim:output:06sequential_11/lstm_38/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitа
-sequential_11/lstm_38/while/lstm_cell/SigmoidSigmoid4sequential_11/lstm_38/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_38/while/lstm_cell/Sigmoid_1Sigmoid4sequential_11/lstm_38/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ┬
)sequential_11/lstm_38/while/lstm_cell/mulMul3sequential_11/lstm_38/while/lstm_cell/Sigmoid_1:y:0)sequential_11_lstm_38_while_placeholder_3*
T0*'
_output_shapes
:          џ
*sequential_11/lstm_38/while/lstm_cell/ReluRelu4sequential_11/lstm_38/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Л
+sequential_11/lstm_38/while/lstm_cell/mul_1Mul1sequential_11/lstm_38/while/lstm_cell/Sigmoid:y:08sequential_11/lstm_38/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          к
+sequential_11/lstm_38/while/lstm_cell/add_1AddV2-sequential_11/lstm_38/while/lstm_cell/mul:z:0/sequential_11/lstm_38/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_38/while/lstm_cell/Sigmoid_2Sigmoid4sequential_11/lstm_38/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:          Ќ
,sequential_11/lstm_38/while/lstm_cell/Relu_1Relu/sequential_11/lstm_38/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Н
+sequential_11/lstm_38/while/lstm_cell/mul_2Mul3sequential_11/lstm_38/while/lstm_cell/Sigmoid_2:y:0:sequential_11/lstm_38/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ѕ
Fsequential_11/lstm_38/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ┬
@sequential_11/lstm_38/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_38_while_placeholder_1Osequential_11/lstm_38/while/TensorArrayV2Write/TensorListSetItem/index:output:0/sequential_11/lstm_38/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_11/lstm_38/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_11/lstm_38/while/addAddV2'sequential_11_lstm_38_while_placeholder*sequential_11/lstm_38/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_11/lstm_38/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_11/lstm_38/while/add_1AddV2Dsequential_11_lstm_38_while_sequential_11_lstm_38_while_loop_counter,sequential_11/lstm_38/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_11/lstm_38/while/IdentityIdentity%sequential_11/lstm_38/while/add_1:z:0!^sequential_11/lstm_38/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_11/lstm_38/while/Identity_1IdentityJsequential_11_lstm_38_while_sequential_11_lstm_38_while_maximum_iterations!^sequential_11/lstm_38/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_11/lstm_38/while/Identity_2Identity#sequential_11/lstm_38/while/add:z:0!^sequential_11/lstm_38/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_11/lstm_38/while/Identity_3IdentityPsequential_11/lstm_38/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_38/while/NoOp*
T0*
_output_shapes
: И
&sequential_11/lstm_38/while/Identity_4Identity/sequential_11/lstm_38/while/lstm_cell/mul_2:z:0!^sequential_11/lstm_38/while/NoOp*
T0*'
_output_shapes
:          И
&sequential_11/lstm_38/while/Identity_5Identity/sequential_11/lstm_38/while/lstm_cell/add_1:z:0!^sequential_11/lstm_38/while/NoOp*
T0*'
_output_shapes
:          ч
 sequential_11/lstm_38/while/NoOpNoOp=^sequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOp<^sequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOp>^sequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "Y
&sequential_11_lstm_38_while_identity_1/sequential_11/lstm_38/while/Identity_1:output:0"Y
&sequential_11_lstm_38_while_identity_2/sequential_11/lstm_38/while/Identity_2:output:0"Y
&sequential_11_lstm_38_while_identity_3/sequential_11/lstm_38/while/Identity_3:output:0"Y
&sequential_11_lstm_38_while_identity_4/sequential_11/lstm_38/while/Identity_4:output:0"Y
&sequential_11_lstm_38_while_identity_5/sequential_11/lstm_38/while/Identity_5:output:0"U
$sequential_11_lstm_38_while_identity-sequential_11/lstm_38/while/Identity:output:0"љ
Esequential_11_lstm_38_while_lstm_cell_biasadd_readvariableop_resourceGsequential_11_lstm_38_while_lstm_cell_biasadd_readvariableop_resource_0"њ
Fsequential_11_lstm_38_while_lstm_cell_matmul_1_readvariableop_resourceHsequential_11_lstm_38_while_lstm_cell_matmul_1_readvariableop_resource_0"ј
Dsequential_11_lstm_38_while_lstm_cell_matmul_readvariableop_resourceFsequential_11_lstm_38_while_lstm_cell_matmul_readvariableop_resource_0"ѕ
Asequential_11_lstm_38_while_sequential_11_lstm_38_strided_slice_1Csequential_11_lstm_38_while_sequential_11_lstm_38_strided_slice_1_0"ђ
}sequential_11_lstm_38_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_38_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_38_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_38_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2|
<sequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOp<sequential_11/lstm_38/while/lstm_cell/BiasAdd/ReadVariableOp2z
;sequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOp;sequential_11/lstm_38/while/lstm_cell/MatMul/ReadVariableOp2~
=sequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOp=sequential_11/lstm_38/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:uq

_output_shapes
: 
W
_user_specified_name?=sequential_11/lstm_38/TensorArrayUnstack/TensorListFromTensor:]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_38/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_38/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_38/while/loop_counter
┬	
├
while_cond_759799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_759799___redundant_placeholder04
0while_while_cond_759799___redundant_placeholder14
0while_while_cond_759799___redundant_placeholder24
0while_while_cond_759799___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
є:
х
while_body_764763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
м4
Н
I__inference_sequential_11_layer_call_and_return_conditional_losses_762460
lstm_35_input!
lstm_35_761960:	ђ!
lstm_35_761962:	 ђ
lstm_35_761964:	ђ!
lstm_36_762116:	 ђ!
lstm_36_762118:	 ђ
lstm_36_762120:	ђ!
lstm_37_762272:	 ђ!
lstm_37_762274:	 ђ
lstm_37_762276:	ђ!
lstm_38_762430:	 ђ!
lstm_38_762432:	 ђ
lstm_38_762434:	ђ!
dense_31_762443:  
dense_31_762445: !
dense_32_762454: 
dense_32_762456:
identityѕб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallбlstm_35/StatefulPartitionedCallбlstm_36/StatefulPartitionedCallбlstm_37/StatefulPartitionedCallбlstm_38/StatefulPartitionedCallЅ
lstm_35/StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputlstm_35_761960lstm_35_761962lstm_35_761964*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_761959Р
dropout_55/PartitionedCallPartitionedCall(lstm_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_761971Ъ
lstm_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_55/PartitionedCall:output:0lstm_36_762116lstm_36_762118lstm_36_762120*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_762115Р
dropout_56/PartitionedCallPartitionedCall(lstm_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_56_layer_call_and_return_conditional_losses_762127Ъ
lstm_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_56/PartitionedCall:output:0lstm_37_762272lstm_37_762274lstm_37_762276*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_762271Р
dropout_57/PartitionedCallPartitionedCall(lstm_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_57_layer_call_and_return_conditional_losses_762283Џ
lstm_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_57/PartitionedCall:output:0lstm_38_762430lstm_38_762432lstm_38_762434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_762429я
dropout_58/PartitionedCallPartitionedCall(lstm_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_58_layer_call_and_return_conditional_losses_762441Ї
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0dense_31_762443dense_31_762445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_761779▀
dropout_59/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_762452Ї
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_59/PartitionedCall:output:0dense_32_762454dense_32_762456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_761807x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ­
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall ^lstm_38/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall2B
lstm_38/StatefulPartitionedCalllstm_38/StatefulPartitionedCall:&"
 
_user_specified_name762456:&"
 
_user_specified_name762454:&"
 
_user_specified_name762445:&"
 
_user_specified_name762443:&"
 
_user_specified_name762434:&"
 
_user_specified_name762432:&
"
 
_user_specified_name762430:&	"
 
_user_specified_name762276:&"
 
_user_specified_name762274:&"
 
_user_specified_name762272:&"
 
_user_specified_name762120:&"
 
_user_specified_name762118:&"
 
_user_specified_name762116:&"
 
_user_specified_name761964:&"
 
_user_specified_name761962:&"
 
_user_specified_name761960:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
»
х
.__inference_sequential_11_layer_call_fn_762497
lstm_35_input
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
	unknown_2:	 ђ
	unknown_3:	 ђ
	unknown_4:	ђ
	unknown_5:	 ђ
	unknown_6:	 ђ
	unknown_7:	ђ
	unknown_8:	 ђ
	unknown_9:	 ђ

unknown_10:	ђ

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_761814o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762493:&"
 
_user_specified_name762491:&"
 
_user_specified_name762489:&"
 
_user_specified_name762487:&"
 
_user_specified_name762485:&"
 
_user_specified_name762483:&
"
 
_user_specified_name762481:&	"
 
_user_specified_name762479:&"
 
_user_specified_name762477:&"
 
_user_specified_name762475:&"
 
_user_specified_name762473:&"
 
_user_specified_name762471:&"
 
_user_specified_name762469:&"
 
_user_specified_name762467:&"
 
_user_specified_name762465:&"
 
_user_specified_name762463:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
┬	
├
while_cond_760290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760290___redundant_placeholder04
0while_while_cond_760290___redundant_placeholder14
0while_while_cond_760290___redundant_placeholder24
0while_while_cond_760290___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
├Б
╬4
__inference__traced_save_766114
file_prefix8
&read_disablecopyonread_dense_31_kernel:  4
&read_1_disablecopyonread_dense_31_bias: :
(read_2_disablecopyonread_dense_32_kernel: 4
&read_3_disablecopyonread_dense_32_bias:D
1read_4_disablecopyonread_lstm_35_lstm_cell_kernel:	ђN
;read_5_disablecopyonread_lstm_35_lstm_cell_recurrent_kernel:	 ђ>
/read_6_disablecopyonread_lstm_35_lstm_cell_bias:	ђD
1read_7_disablecopyonread_lstm_36_lstm_cell_kernel:	 ђN
;read_8_disablecopyonread_lstm_36_lstm_cell_recurrent_kernel:	 ђ>
/read_9_disablecopyonread_lstm_36_lstm_cell_bias:	ђE
2read_10_disablecopyonread_lstm_37_lstm_cell_kernel:	 ђO
<read_11_disablecopyonread_lstm_37_lstm_cell_recurrent_kernel:	 ђ?
0read_12_disablecopyonread_lstm_37_lstm_cell_bias:	ђE
2read_13_disablecopyonread_lstm_38_lstm_cell_kernel:	 ђO
<read_14_disablecopyonread_lstm_38_lstm_cell_recurrent_kernel:	 ђ?
0read_15_disablecopyonread_lstm_38_lstm_cell_bias:	ђ-
#read_16_disablecopyonread_iteration:	 1
'read_17_disablecopyonread_learning_rate: L
9read_18_disablecopyonread_adam_m_lstm_35_lstm_cell_kernel:	ђL
9read_19_disablecopyonread_adam_v_lstm_35_lstm_cell_kernel:	ђV
Cread_20_disablecopyonread_adam_m_lstm_35_lstm_cell_recurrent_kernel:	 ђV
Cread_21_disablecopyonread_adam_v_lstm_35_lstm_cell_recurrent_kernel:	 ђF
7read_22_disablecopyonread_adam_m_lstm_35_lstm_cell_bias:	ђF
7read_23_disablecopyonread_adam_v_lstm_35_lstm_cell_bias:	ђL
9read_24_disablecopyonread_adam_m_lstm_36_lstm_cell_kernel:	 ђL
9read_25_disablecopyonread_adam_v_lstm_36_lstm_cell_kernel:	 ђV
Cread_26_disablecopyonread_adam_m_lstm_36_lstm_cell_recurrent_kernel:	 ђV
Cread_27_disablecopyonread_adam_v_lstm_36_lstm_cell_recurrent_kernel:	 ђF
7read_28_disablecopyonread_adam_m_lstm_36_lstm_cell_bias:	ђF
7read_29_disablecopyonread_adam_v_lstm_36_lstm_cell_bias:	ђL
9read_30_disablecopyonread_adam_m_lstm_37_lstm_cell_kernel:	 ђL
9read_31_disablecopyonread_adam_v_lstm_37_lstm_cell_kernel:	 ђV
Cread_32_disablecopyonread_adam_m_lstm_37_lstm_cell_recurrent_kernel:	 ђV
Cread_33_disablecopyonread_adam_v_lstm_37_lstm_cell_recurrent_kernel:	 ђF
7read_34_disablecopyonread_adam_m_lstm_37_lstm_cell_bias:	ђF
7read_35_disablecopyonread_adam_v_lstm_37_lstm_cell_bias:	ђL
9read_36_disablecopyonread_adam_m_lstm_38_lstm_cell_kernel:	 ђL
9read_37_disablecopyonread_adam_v_lstm_38_lstm_cell_kernel:	 ђV
Cread_38_disablecopyonread_adam_m_lstm_38_lstm_cell_recurrent_kernel:	 ђV
Cread_39_disablecopyonread_adam_v_lstm_38_lstm_cell_recurrent_kernel:	 ђF
7read_40_disablecopyonread_adam_m_lstm_38_lstm_cell_bias:	ђF
7read_41_disablecopyonread_adam_v_lstm_38_lstm_cell_bias:	ђB
0read_42_disablecopyonread_adam_m_dense_31_kernel:  B
0read_43_disablecopyonread_adam_v_dense_31_kernel:  <
.read_44_disablecopyonread_adam_m_dense_31_bias: <
.read_45_disablecopyonread_adam_v_dense_31_bias: B
0read_46_disablecopyonread_adam_m_dense_32_kernel: B
0read_47_disablecopyonread_adam_v_dense_32_kernel: <
.read_48_disablecopyonread_adam_m_dense_32_bias:<
.read_49_disablecopyonread_adam_v_dense_32_bias:+
!read_50_disablecopyonread_total_1: +
!read_51_disablecopyonread_count_1: )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: 
savev2_const
identity_109ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_24/DisableCopyOnReadбRead_24/ReadVariableOpбRead_25/DisableCopyOnReadбRead_25/ReadVariableOpбRead_26/DisableCopyOnReadбRead_26/ReadVariableOpбRead_27/DisableCopyOnReadбRead_27/ReadVariableOpбRead_28/DisableCopyOnReadбRead_28/ReadVariableOpбRead_29/DisableCopyOnReadбRead_29/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_30/DisableCopyOnReadбRead_30/ReadVariableOpбRead_31/DisableCopyOnReadбRead_31/ReadVariableOpбRead_32/DisableCopyOnReadбRead_32/ReadVariableOpбRead_33/DisableCopyOnReadбRead_33/ReadVariableOpбRead_34/DisableCopyOnReadбRead_34/ReadVariableOpбRead_35/DisableCopyOnReadбRead_35/ReadVariableOpбRead_36/DisableCopyOnReadбRead_36/ReadVariableOpбRead_37/DisableCopyOnReadбRead_37/ReadVariableOpбRead_38/DisableCopyOnReadбRead_38/ReadVariableOpбRead_39/DisableCopyOnReadбRead_39/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_40/DisableCopyOnReadбRead_40/ReadVariableOpбRead_41/DisableCopyOnReadбRead_41/ReadVariableOpбRead_42/DisableCopyOnReadбRead_42/ReadVariableOpбRead_43/DisableCopyOnReadбRead_43/ReadVariableOpбRead_44/DisableCopyOnReadбRead_44/ReadVariableOpбRead_45/DisableCopyOnReadбRead_45/ReadVariableOpбRead_46/DisableCopyOnReadбRead_46/ReadVariableOpбRead_47/DisableCopyOnReadбRead_47/ReadVariableOpбRead_48/DisableCopyOnReadбRead_48/ReadVariableOpбRead_49/DisableCopyOnReadбRead_49/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_50/DisableCopyOnReadбRead_50/ReadVariableOpбRead_51/DisableCopyOnReadбRead_51/ReadVariableOpбRead_52/DisableCopyOnReadбRead_52/ReadVariableOpбRead_53/DisableCopyOnReadбRead_53/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_31_kernel"/device:CPU:0*
_output_shapes
 б
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_31_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:  z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_31_bias"/device:CPU:0*
_output_shapes
 б
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_31_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_32_kernel"/device:CPU:0*
_output_shapes
 е
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_32_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

: z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_32_bias"/device:CPU:0*
_output_shapes
 б
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_32_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:Ё
Read_4/DisableCopyOnReadDisableCopyOnRead1read_4_disablecopyonread_lstm_35_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_4/ReadVariableOpReadVariableOp1read_4_disablecopyonread_lstm_35_lstm_cell_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђd

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЈ
Read_5/DisableCopyOnReadDisableCopyOnRead;read_5_disablecopyonread_lstm_35_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_5/ReadVariableOpReadVariableOp;read_5_disablecopyonread_lstm_35_lstm_cell_recurrent_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЃ
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_lstm_35_lstm_cell_bias"/device:CPU:0*
_output_shapes
 г
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_lstm_35_lstm_cell_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЁ
Read_7/DisableCopyOnReadDisableCopyOnRead1read_7_disablecopyonread_lstm_36_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_7/ReadVariableOpReadVariableOp1read_7_disablecopyonread_lstm_36_lstm_cell_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЈ
Read_8/DisableCopyOnReadDisableCopyOnRead;read_8_disablecopyonread_lstm_36_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_8/ReadVariableOpReadVariableOp;read_8_disablecopyonread_lstm_36_lstm_cell_recurrent_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЃ
Read_9/DisableCopyOnReadDisableCopyOnRead/read_9_disablecopyonread_lstm_36_lstm_cell_bias"/device:CPU:0*
_output_shapes
 г
Read_9/ReadVariableOpReadVariableOp/read_9_disablecopyonread_lstm_36_lstm_cell_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЄ
Read_10/DisableCopyOnReadDisableCopyOnRead2read_10_disablecopyonread_lstm_37_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 х
Read_10/ReadVariableOpReadVariableOp2read_10_disablecopyonread_lstm_37_lstm_cell_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЉ
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_lstm_37_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_lstm_37_lstm_cell_recurrent_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЁ
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_lstm_37_lstm_cell_bias"/device:CPU:0*
_output_shapes
 »
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_lstm_37_lstm_cell_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЄ
Read_13/DisableCopyOnReadDisableCopyOnRead2read_13_disablecopyonread_lstm_38_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 х
Read_13/ReadVariableOpReadVariableOp2read_13_disablecopyonread_lstm_38_lstm_cell_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЉ
Read_14/DisableCopyOnReadDisableCopyOnRead<read_14_disablecopyonread_lstm_38_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_14/ReadVariableOpReadVariableOp<read_14_disablecopyonread_lstm_38_lstm_cell_recurrent_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђЁ
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_lstm_38_lstm_cell_bias"/device:CPU:0*
_output_shapes
 »
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_lstm_38_lstm_cell_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђx
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ю
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_iteration^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 А
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_learning_rate^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: ј
Read_18/DisableCopyOnReadDisableCopyOnRead9read_18_disablecopyonread_adam_m_lstm_35_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_18/ReadVariableOpReadVariableOp9read_18_disablecopyonread_adam_m_lstm_35_lstm_cell_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђј
Read_19/DisableCopyOnReadDisableCopyOnRead9read_19_disablecopyonread_adam_v_lstm_35_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_19/ReadVariableOpReadVariableOp9read_19_disablecopyonread_adam_v_lstm_35_lstm_cell_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђў
Read_20/DisableCopyOnReadDisableCopyOnReadCread_20_disablecopyonread_adam_m_lstm_35_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_20/ReadVariableOpReadVariableOpCread_20_disablecopyonread_adam_m_lstm_35_lstm_cell_recurrent_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_21/DisableCopyOnReadDisableCopyOnReadCread_21_disablecopyonread_adam_v_lstm_35_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_21/ReadVariableOpReadVariableOpCread_21_disablecopyonread_adam_v_lstm_35_lstm_cell_recurrent_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђї
Read_22/DisableCopyOnReadDisableCopyOnRead7read_22_disablecopyonread_adam_m_lstm_35_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_22/ReadVariableOpReadVariableOp7read_22_disablecopyonread_adam_m_lstm_35_lstm_cell_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђї
Read_23/DisableCopyOnReadDisableCopyOnRead7read_23_disablecopyonread_adam_v_lstm_35_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_23/ReadVariableOpReadVariableOp7read_23_disablecopyonread_adam_v_lstm_35_lstm_cell_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђј
Read_24/DisableCopyOnReadDisableCopyOnRead9read_24_disablecopyonread_adam_m_lstm_36_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_24/ReadVariableOpReadVariableOp9read_24_disablecopyonread_adam_m_lstm_36_lstm_cell_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђј
Read_25/DisableCopyOnReadDisableCopyOnRead9read_25_disablecopyonread_adam_v_lstm_36_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_25/ReadVariableOpReadVariableOp9read_25_disablecopyonread_adam_v_lstm_36_lstm_cell_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_26/DisableCopyOnReadDisableCopyOnReadCread_26_disablecopyonread_adam_m_lstm_36_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_26/ReadVariableOpReadVariableOpCread_26_disablecopyonread_adam_m_lstm_36_lstm_cell_recurrent_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_27/DisableCopyOnReadDisableCopyOnReadCread_27_disablecopyonread_adam_v_lstm_36_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_27/ReadVariableOpReadVariableOpCread_27_disablecopyonread_adam_v_lstm_36_lstm_cell_recurrent_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђї
Read_28/DisableCopyOnReadDisableCopyOnRead7read_28_disablecopyonread_adam_m_lstm_36_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_28/ReadVariableOpReadVariableOp7read_28_disablecopyonread_adam_m_lstm_36_lstm_cell_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђї
Read_29/DisableCopyOnReadDisableCopyOnRead7read_29_disablecopyonread_adam_v_lstm_36_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_29/ReadVariableOpReadVariableOp7read_29_disablecopyonread_adam_v_lstm_36_lstm_cell_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђј
Read_30/DisableCopyOnReadDisableCopyOnRead9read_30_disablecopyonread_adam_m_lstm_37_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_30/ReadVariableOpReadVariableOp9read_30_disablecopyonread_adam_m_lstm_37_lstm_cell_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђј
Read_31/DisableCopyOnReadDisableCopyOnRead9read_31_disablecopyonread_adam_v_lstm_37_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_31/ReadVariableOpReadVariableOp9read_31_disablecopyonread_adam_v_lstm_37_lstm_cell_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_32/DisableCopyOnReadDisableCopyOnReadCread_32_disablecopyonread_adam_m_lstm_37_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_32/ReadVariableOpReadVariableOpCread_32_disablecopyonread_adam_m_lstm_37_lstm_cell_recurrent_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_33/DisableCopyOnReadDisableCopyOnReadCread_33_disablecopyonread_adam_v_lstm_37_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_33/ReadVariableOpReadVariableOpCread_33_disablecopyonread_adam_v_lstm_37_lstm_cell_recurrent_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђї
Read_34/DisableCopyOnReadDisableCopyOnRead7read_34_disablecopyonread_adam_m_lstm_37_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_34/ReadVariableOpReadVariableOp7read_34_disablecopyonread_adam_m_lstm_37_lstm_cell_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђї
Read_35/DisableCopyOnReadDisableCopyOnRead7read_35_disablecopyonread_adam_v_lstm_37_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_35/ReadVariableOpReadVariableOp7read_35_disablecopyonread_adam_v_lstm_37_lstm_cell_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђј
Read_36/DisableCopyOnReadDisableCopyOnRead9read_36_disablecopyonread_adam_m_lstm_38_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_36/ReadVariableOpReadVariableOp9read_36_disablecopyonread_adam_m_lstm_38_lstm_cell_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђј
Read_37/DisableCopyOnReadDisableCopyOnRead9read_37_disablecopyonread_adam_v_lstm_38_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_37/ReadVariableOpReadVariableOp9read_37_disablecopyonread_adam_v_lstm_38_lstm_cell_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_38/DisableCopyOnReadDisableCopyOnReadCread_38_disablecopyonread_adam_m_lstm_38_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_38/ReadVariableOpReadVariableOpCread_38_disablecopyonread_adam_m_lstm_38_lstm_cell_recurrent_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђў
Read_39/DisableCopyOnReadDisableCopyOnReadCread_39_disablecopyonread_adam_v_lstm_38_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 к
Read_39/ReadVariableOpReadVariableOpCread_39_disablecopyonread_adam_v_lstm_38_lstm_cell_recurrent_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 ђ*
dtype0p
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 ђf
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ђї
Read_40/DisableCopyOnReadDisableCopyOnRead7read_40_disablecopyonread_adam_m_lstm_38_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_40/ReadVariableOpReadVariableOp7read_40_disablecopyonread_adam_m_lstm_38_lstm_cell_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђї
Read_41/DisableCopyOnReadDisableCopyOnRead7read_41_disablecopyonread_adam_v_lstm_38_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Х
Read_41/ReadVariableOpReadVariableOp7read_41_disablecopyonread_adam_v_lstm_38_lstm_cell_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЁ
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_dense_31_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_dense_31_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:  Ё
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_dense_31_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_dense_31_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:  Ѓ
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_dense_31_bias"/device:CPU:0*
_output_shapes
 г
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_dense_31_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: Ѓ
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_dense_31_bias"/device:CPU:0*
_output_shapes
 г
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_dense_31_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: Ё
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_dense_32_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_dense_32_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

: Ё
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_dense_32_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_dense_32_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

: Ѓ
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_m_dense_32_bias"/device:CPU:0*
_output_shapes
 г
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_m_dense_32_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_49/DisableCopyOnReadDisableCopyOnRead.read_49_disablecopyonread_adam_v_dense_32_bias"/device:CPU:0*
_output_shapes
 г
Read_49/ReadVariableOpReadVariableOp.read_49_disablecopyonread_adam_v_dense_32_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Џ
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_1^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Џ
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ў
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ў
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: ■
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Д
valueЮBџ7B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▄
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ђ
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ▒
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: ═
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_109Identity_109:output:0*(
_construction_contextkEagerRuntime*Ѓ
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=79

_output_shapes
: 

_user_specified_nameConst:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:'4#
!
_user_specified_name	count_1:'3#
!
_user_specified_name	total_1:420
.
_user_specified_nameAdam/v/dense_32/bias:410
.
_user_specified_nameAdam/m/dense_32/bias:602
0
_user_specified_nameAdam/v/dense_32/kernel:6/2
0
_user_specified_nameAdam/m/dense_32/kernel:4.0
.
_user_specified_nameAdam/v/dense_31/bias:4-0
.
_user_specified_nameAdam/m/dense_31/bias:6,2
0
_user_specified_nameAdam/v/dense_31/kernel:6+2
0
_user_specified_nameAdam/m/dense_31/kernel:=*9
7
_user_specified_nameAdam/v/lstm_38/lstm_cell/bias:=)9
7
_user_specified_nameAdam/m/lstm_38/lstm_cell/bias:I(E
C
_user_specified_name+)Adam/v/lstm_38/lstm_cell/recurrent_kernel:I'E
C
_user_specified_name+)Adam/m/lstm_38/lstm_cell/recurrent_kernel:?&;
9
_user_specified_name!Adam/v/lstm_38/lstm_cell/kernel:?%;
9
_user_specified_name!Adam/m/lstm_38/lstm_cell/kernel:=$9
7
_user_specified_nameAdam/v/lstm_37/lstm_cell/bias:=#9
7
_user_specified_nameAdam/m/lstm_37/lstm_cell/bias:I"E
C
_user_specified_name+)Adam/v/lstm_37/lstm_cell/recurrent_kernel:I!E
C
_user_specified_name+)Adam/m/lstm_37/lstm_cell/recurrent_kernel:? ;
9
_user_specified_name!Adam/v/lstm_37/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_37/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_36/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_36/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_36/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_36/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_36/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_36/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_35/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_35/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_35/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_35/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_35/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_35/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:62
0
_user_specified_namelstm_38/lstm_cell/bias:B>
<
_user_specified_name$"lstm_38/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_38/lstm_cell/kernel:62
0
_user_specified_namelstm_37/lstm_cell/bias:B>
<
_user_specified_name$"lstm_37/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_37/lstm_cell/kernel:6
2
0
_user_specified_namelstm_36/lstm_cell/bias:B	>
<
_user_specified_name$"lstm_36/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_36/lstm_cell/kernel:62
0
_user_specified_namelstm_35/lstm_cell/bias:B>
<
_user_specified_name$"lstm_35/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_35/lstm_cell/kernel:-)
'
_user_specified_namedense_32/bias:/+
)
_user_specified_namedense_32/kernel:-)
'
_user_specified_namedense_31/bias:/+
)
_user_specified_namedense_31/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┬	
├
while_cond_762343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_762343___redundant_placeholder04
0while_while_cond_762343___redundant_placeholder14
0while_while_cond_762343___redundant_placeholder24
0while_while_cond_762343___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_760491
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760491___redundant_placeholder04
0while_while_cond_760491___redundant_placeholder14
0while_while_cond_760491___redundant_placeholder24
0while_while_cond_760491___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╣
з
*__inference_lstm_cell_layer_call_fn_765491

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765483:&"
 
_user_specified_name765481:&"
 
_user_specified_name765479:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
¤
d
+__inference_dropout_59_layer_call_fn_765335

inputs
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_761796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╣
з
*__inference_lstm_cell_layer_call_fn_765687

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760824o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765679:&"
 
_user_specified_name765677:&"
 
_user_specified_name765675:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Х

e
F__inference_dropout_56_layer_call_and_return_conditional_losses_764011

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ж8
х
while_body_763905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_761335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_761335___redundant_placeholder04
0while_while_cond_761335___redundant_placeholder14
0while_while_cond_761335___redundant_placeholder24
0while_while_cond_761335___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_763904
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763904___redundant_placeholder04
0while_while_cond_763904___redundant_placeholder14
0while_while_cond_763904___redundant_placeholder24
0while_while_cond_763904___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
лJ
Ѕ
C__inference_lstm_38_layer_call_and_return_conditional_losses_765138

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_765053*
condR
while_cond_765052*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
џ

e
F__inference_dropout_58_layer_call_and_return_conditional_losses_761767

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:          a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ь
ќ
)__inference_dense_32_layer_call_fn_765366

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_761807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765362:&"
 
_user_specified_name765360:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ж8
х
while_body_764405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Б9
щ
C__inference_lstm_38_layer_call_and_return_conditional_losses_761056

inputs#
lstm_cell_760972:	 ђ#
lstm_cell_760974:	 ђ
lstm_cell_760976:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760972lstm_cell_760974lstm_cell_760976*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760971n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760972lstm_cell_760974lstm_cell_760976*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760986*
condR
while_cond_760985*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760976:&"
 
_user_specified_name760974:&"
 
_user_specified_name760972:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
М
х
(__inference_lstm_36_layer_call_fn_763406

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_761420s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name763402:&"
 
_user_specified_name763400:&"
 
_user_specified_name763398:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760277

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬	
├
while_cond_762030
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_762030___redundant_placeholder04
0while_while_cond_762030___redundant_placeholder14
0while_while_cond_762030___redundant_placeholder24
0while_while_cond_762030___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
§
и
(__inference_lstm_36_layer_call_fn_763395
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_760360|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name763391:&"
 
_user_specified_name763389:&"
 
_user_specified_name763387:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
┬	
├
while_cond_764547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764547___redundant_placeholder04
0while_while_cond_764547___redundant_placeholder14
0while_while_cond_764547___redundant_placeholder24
0while_while_cond_764547___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_37_layer_call_and_return_conditional_losses_761583

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_761499*
condR
while_cond_761498*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
зJ
І
C__inference_lstm_38_layer_call_and_return_conditional_losses_764993
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764908*
condR
while_cond_764907*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
╦
х
(__inference_lstm_38_layer_call_fn_764703

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_762429o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764699:&"
 
_user_specified_name764697:&"
 
_user_specified_name764695:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Б9
щ
C__inference_lstm_38_layer_call_and_return_conditional_losses_760909

inputs#
lstm_cell_760825:	 ђ#
lstm_cell_760827:	 ђ
lstm_cell_760829:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760825lstm_cell_760827lstm_cell_760829*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760824n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760825lstm_cell_760827lstm_cell_760829*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760839*
condR
while_cond_760838*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760829:&"
 
_user_specified_name760827:&"
 
_user_specified_name760825:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
┬	
├
while_cond_764118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764118___redundant_placeholder04
0while_while_cond_764118___redundant_placeholder14
0while_while_cond_764118___redundant_placeholder24
0while_while_cond_764118___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┘
d
F__inference_dropout_58_layer_call_and_return_conditional_losses_765310

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760623

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
лJ
Ѕ
C__inference_lstm_38_layer_call_and_return_conditional_losses_762429

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_762344*
condR
while_cond_762343*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
§
и
(__inference_lstm_37_layer_call_fn_764038
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_760706|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764034:&"
 
_user_specified_name764032:&"
 
_user_specified_name764030:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
╦

ш
D__inference_dense_31_layer_call_and_return_conditional_losses_765330

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╩I
Ѕ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763846

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763762*
condR
while_cond_763761*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
┬	
├
while_cond_765197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_765197___redundant_placeholder04
0while_while_cond_765197___redundant_placeholder14
0while_while_cond_765197___redundant_placeholder24
0while_while_cond_765197___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▀
d
+__inference_dropout_57_layer_call_fn_764637

inputs
identityѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_57_layer_call_and_return_conditional_losses_761602s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760971

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ж8
х
while_body_763476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ѕJ
І
C__inference_lstm_36_layer_call_and_return_conditional_losses_763560
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763476*
condR
while_cond_763475*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
ь
ч
'sequential_11_lstm_35_while_cond_759202H
Dsequential_11_lstm_35_while_sequential_11_lstm_35_while_loop_counterN
Jsequential_11_lstm_35_while_sequential_11_lstm_35_while_maximum_iterations+
'sequential_11_lstm_35_while_placeholder-
)sequential_11_lstm_35_while_placeholder_1-
)sequential_11_lstm_35_while_placeholder_2-
)sequential_11_lstm_35_while_placeholder_3J
Fsequential_11_lstm_35_while_less_sequential_11_lstm_35_strided_slice_1`
\sequential_11_lstm_35_while_sequential_11_lstm_35_while_cond_759202___redundant_placeholder0`
\sequential_11_lstm_35_while_sequential_11_lstm_35_while_cond_759202___redundant_placeholder1`
\sequential_11_lstm_35_while_sequential_11_lstm_35_while_cond_759202___redundant_placeholder2`
\sequential_11_lstm_35_while_sequential_11_lstm_35_while_cond_759202___redundant_placeholder3(
$sequential_11_lstm_35_while_identity
║
 sequential_11/lstm_35/while/LessLess'sequential_11_lstm_35_while_placeholderFsequential_11_lstm_35_while_less_sequential_11_lstm_35_strided_slice_1*
T0*
_output_shapes
: w
$sequential_11/lstm_35/while/IdentityIdentity$sequential_11/lstm_35/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_11_lstm_35_while_identity-sequential_11/lstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_35/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_35/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_35/while/loop_counter
┬	
├
while_cond_763261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763261___redundant_placeholder04
0while_while_cond_763261___redundant_placeholder14
0while_while_cond_763261___redundant_placeholder24
0while_while_cond_763261___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╣
з
*__inference_lstm_cell_layer_call_fn_765508

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765500:&"
 
_user_specified_name765498:&"
 
_user_specified_name765496:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
§
и
(__inference_lstm_35_layer_call_fn_762741
inputs_0
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_759869|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762737:&"
 
_user_specified_name762735:&"
 
_user_specified_name762733:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
ИS
э
'sequential_11_lstm_36_while_body_759343H
Dsequential_11_lstm_36_while_sequential_11_lstm_36_while_loop_counterN
Jsequential_11_lstm_36_while_sequential_11_lstm_36_while_maximum_iterations+
'sequential_11_lstm_36_while_placeholder-
)sequential_11_lstm_36_while_placeholder_1-
)sequential_11_lstm_36_while_placeholder_2-
)sequential_11_lstm_36_while_placeholder_3G
Csequential_11_lstm_36_while_sequential_11_lstm_36_strided_slice_1_0Ѓ
sequential_11_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_36_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_11_lstm_36_while_lstm_cell_matmul_readvariableop_resource_0:	 ђ[
Hsequential_11_lstm_36_while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђV
Gsequential_11_lstm_36_while_lstm_cell_biasadd_readvariableop_resource_0:	ђ(
$sequential_11_lstm_36_while_identity*
&sequential_11_lstm_36_while_identity_1*
&sequential_11_lstm_36_while_identity_2*
&sequential_11_lstm_36_while_identity_3*
&sequential_11_lstm_36_while_identity_4*
&sequential_11_lstm_36_while_identity_5E
Asequential_11_lstm_36_while_sequential_11_lstm_36_strided_slice_1Ђ
}sequential_11_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_36_tensorarrayunstack_tensorlistfromtensorW
Dsequential_11_lstm_36_while_lstm_cell_matmul_readvariableop_resource:	 ђY
Fsequential_11_lstm_36_while_lstm_cell_matmul_1_readvariableop_resource:	 ђT
Esequential_11_lstm_36_while_lstm_cell_biasadd_readvariableop_resource:	ђѕб<sequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOpб;sequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOpб=sequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOpъ
Msequential_11/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ћ
?sequential_11/lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_36_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_36_while_placeholderVsequential_11/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0├
;sequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpFsequential_11_lstm_36_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Ш
,sequential_11/lstm_36/while/lstm_cell/MatMulMatMulFsequential_11/lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђК
=sequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHsequential_11_lstm_36_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0П
.sequential_11/lstm_36/while/lstm_cell/MatMul_1MatMul)sequential_11_lstm_36_while_placeholder_2Esequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђО
)sequential_11/lstm_36/while/lstm_cell/addAddV26sequential_11/lstm_36/while/lstm_cell/MatMul:product:08sequential_11/lstm_36/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ┴
<sequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGsequential_11_lstm_36_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0Я
-sequential_11/lstm_36/while/lstm_cell/BiasAddBiasAdd-sequential_11/lstm_36/while/lstm_cell/add:z:0Dsequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
5sequential_11/lstm_36/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :е
+sequential_11/lstm_36/while/lstm_cell/splitSplit>sequential_11/lstm_36/while/lstm_cell/split/split_dim:output:06sequential_11/lstm_36/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitа
-sequential_11/lstm_36/while/lstm_cell/SigmoidSigmoid4sequential_11/lstm_36/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_36/while/lstm_cell/Sigmoid_1Sigmoid4sequential_11/lstm_36/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ┬
)sequential_11/lstm_36/while/lstm_cell/mulMul3sequential_11/lstm_36/while/lstm_cell/Sigmoid_1:y:0)sequential_11_lstm_36_while_placeholder_3*
T0*'
_output_shapes
:          џ
*sequential_11/lstm_36/while/lstm_cell/ReluRelu4sequential_11/lstm_36/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Л
+sequential_11/lstm_36/while/lstm_cell/mul_1Mul1sequential_11/lstm_36/while/lstm_cell/Sigmoid:y:08sequential_11/lstm_36/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          к
+sequential_11/lstm_36/while/lstm_cell/add_1AddV2-sequential_11/lstm_36/while/lstm_cell/mul:z:0/sequential_11/lstm_36/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_36/while/lstm_cell/Sigmoid_2Sigmoid4sequential_11/lstm_36/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:          Ќ
,sequential_11/lstm_36/while/lstm_cell/Relu_1Relu/sequential_11/lstm_36/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Н
+sequential_11/lstm_36/while/lstm_cell/mul_2Mul3sequential_11/lstm_36/while/lstm_cell/Sigmoid_2:y:0:sequential_11/lstm_36/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          џ
@sequential_11/lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_36_while_placeholder_1'sequential_11_lstm_36_while_placeholder/sequential_11/lstm_36/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_11/lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_11/lstm_36/while/addAddV2'sequential_11_lstm_36_while_placeholder*sequential_11/lstm_36/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_11/lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_11/lstm_36/while/add_1AddV2Dsequential_11_lstm_36_while_sequential_11_lstm_36_while_loop_counter,sequential_11/lstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_11/lstm_36/while/IdentityIdentity%sequential_11/lstm_36/while/add_1:z:0!^sequential_11/lstm_36/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_11/lstm_36/while/Identity_1IdentityJsequential_11_lstm_36_while_sequential_11_lstm_36_while_maximum_iterations!^sequential_11/lstm_36/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_11/lstm_36/while/Identity_2Identity#sequential_11/lstm_36/while/add:z:0!^sequential_11/lstm_36/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_11/lstm_36/while/Identity_3IdentityPsequential_11/lstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_36/while/NoOp*
T0*
_output_shapes
: И
&sequential_11/lstm_36/while/Identity_4Identity/sequential_11/lstm_36/while/lstm_cell/mul_2:z:0!^sequential_11/lstm_36/while/NoOp*
T0*'
_output_shapes
:          И
&sequential_11/lstm_36/while/Identity_5Identity/sequential_11/lstm_36/while/lstm_cell/add_1:z:0!^sequential_11/lstm_36/while/NoOp*
T0*'
_output_shapes
:          ч
 sequential_11/lstm_36/while/NoOpNoOp=^sequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOp<^sequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOp>^sequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "Y
&sequential_11_lstm_36_while_identity_1/sequential_11/lstm_36/while/Identity_1:output:0"Y
&sequential_11_lstm_36_while_identity_2/sequential_11/lstm_36/while/Identity_2:output:0"Y
&sequential_11_lstm_36_while_identity_3/sequential_11/lstm_36/while/Identity_3:output:0"Y
&sequential_11_lstm_36_while_identity_4/sequential_11/lstm_36/while/Identity_4:output:0"Y
&sequential_11_lstm_36_while_identity_5/sequential_11/lstm_36/while/Identity_5:output:0"U
$sequential_11_lstm_36_while_identity-sequential_11/lstm_36/while/Identity:output:0"љ
Esequential_11_lstm_36_while_lstm_cell_biasadd_readvariableop_resourceGsequential_11_lstm_36_while_lstm_cell_biasadd_readvariableop_resource_0"њ
Fsequential_11_lstm_36_while_lstm_cell_matmul_1_readvariableop_resourceHsequential_11_lstm_36_while_lstm_cell_matmul_1_readvariableop_resource_0"ј
Dsequential_11_lstm_36_while_lstm_cell_matmul_readvariableop_resourceFsequential_11_lstm_36_while_lstm_cell_matmul_readvariableop_resource_0"ѕ
Asequential_11_lstm_36_while_sequential_11_lstm_36_strided_slice_1Csequential_11_lstm_36_while_sequential_11_lstm_36_strided_slice_1_0"ђ
}sequential_11_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_36_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2|
<sequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOp<sequential_11/lstm_36/while/lstm_cell/BiasAdd/ReadVariableOp2z
;sequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOp;sequential_11/lstm_36/while/lstm_cell/MatMul/ReadVariableOp2~
=sequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOp=sequential_11/lstm_36/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:uq

_output_shapes
: 
W
_user_specified_name?=sequential_11/lstm_36/TensorArrayUnstack/TensorListFromTensor:]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_36/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_36/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_36/while/loop_counter
с
и
(__inference_lstm_38_layer_call_fn_764681
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_761056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764677:&"
 
_user_specified_name764675:&"
 
_user_specified_name764673:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
┬	
├
while_cond_762975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_762975___redundant_placeholder04
0while_while_cond_762975___redundant_placeholder14
0while_while_cond_762975___redundant_placeholder24
0while_while_cond_762975___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╣
з
*__inference_lstm_cell_layer_call_fn_765393

inputs
states_0
states_1
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765385:&"
 
_user_specified_name765383:&"
 
_user_specified_name765381:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є:
х
while_body_762344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╦
х
(__inference_lstm_38_layer_call_fn_764692

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_761748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764688:&"
 
_user_specified_name764686:&"
 
_user_specified_name764684:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ж8
х
while_body_761875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И8
щ
C__inference_lstm_35_layer_call_and_return_conditional_losses_760014

inputs#
lstm_cell_759932:	ђ#
lstm_cell_759934:	 ђ
lstm_cell_759936:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_759932lstm_cell_759934lstm_cell_759936*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759931n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_759932lstm_cell_759934lstm_cell_759936*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_759945*
condR
while_cond_759944*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name759936:&"
 
_user_specified_name759934:&"
 
_user_specified_name759932:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
┬	
├
while_cond_764261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764261___redundant_placeholder04
0while_while_cond_764261___redundant_placeholder14
0while_while_cond_764261___redundant_placeholder24
0while_while_cond_764261___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▀
d
+__inference_dropout_56_layer_call_fn_763994

inputs
identityѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_56_layer_call_and_return_conditional_losses_761439s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
▒
G
+__inference_dropout_57_layer_call_fn_764642

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_57_layer_call_and_return_conditional_losses_762283d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ж8
х
while_body_763262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_36_layer_call_and_return_conditional_losses_761420

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_761336*
condR
while_cond_761335*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765474

inputs
states_0
states_11
matmul_readvariableop_resource:	ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_760636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760636___redundant_placeholder04
0while_while_cond_760636___redundant_placeholder14
0while_while_cond_760636___redundant_placeholder24
0while_while_cond_760636___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ь
ч
'sequential_11_lstm_36_while_cond_759342H
Dsequential_11_lstm_36_while_sequential_11_lstm_36_while_loop_counterN
Jsequential_11_lstm_36_while_sequential_11_lstm_36_while_maximum_iterations+
'sequential_11_lstm_36_while_placeholder-
)sequential_11_lstm_36_while_placeholder_1-
)sequential_11_lstm_36_while_placeholder_2-
)sequential_11_lstm_36_while_placeholder_3J
Fsequential_11_lstm_36_while_less_sequential_11_lstm_36_strided_slice_1`
\sequential_11_lstm_36_while_sequential_11_lstm_36_while_cond_759342___redundant_placeholder0`
\sequential_11_lstm_36_while_sequential_11_lstm_36_while_cond_759342___redundant_placeholder1`
\sequential_11_lstm_36_while_sequential_11_lstm_36_while_cond_759342___redundant_placeholder2`
\sequential_11_lstm_36_while_sequential_11_lstm_36_while_cond_759342___redundant_placeholder3(
$sequential_11_lstm_36_while_identity
║
 sequential_11/lstm_36/while/LessLess'sequential_11_lstm_36_while_placeholderFsequential_11_lstm_36_while_less_sequential_11_lstm_36_strided_slice_1*
T0*
_output_shapes
: w
$sequential_11/lstm_36/while/IdentityIdentity$sequential_11/lstm_36/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_11_lstm_36_while_identity-sequential_11/lstm_36/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_36/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_36/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_36/while/loop_counter
┬	
├
while_cond_761662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_761662___redundant_placeholder04
0while_while_cond_761662___redundant_placeholder14
0while_while_cond_761662___redundant_placeholder24
0while_while_cond_761662___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
М
х
(__inference_lstm_37_layer_call_fn_764049

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_761583s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764045:&"
 
_user_specified_name764043:&"
 
_user_specified_name764041:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
М
х
(__inference_lstm_36_layer_call_fn_763417

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_762115s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name763413:&"
 
_user_specified_name763411:&"
 
_user_specified_name763409:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
┬	
├
while_cond_760985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760985___redundant_placeholder04
0while_while_cond_760985___redundant_placeholder14
0while_while_cond_760985___redundant_placeholder24
0while_while_cond_760985___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ѕJ
І
C__inference_lstm_35_layer_call_and_return_conditional_losses_763060
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_762976*
condR
while_cond_762975*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
§
и
(__inference_lstm_35_layer_call_fn_762752
inputs_0
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_760014|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762748:&"
 
_user_specified_name762746:&"
 
_user_specified_name762744:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
┬$
╬
while_body_759945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_759969_0:	ђ+
while_lstm_cell_759971_0:	 ђ'
while_lstm_cell_759973_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_759969:	ђ)
while_lstm_cell_759971:	 ђ%
while_lstm_cell_759973:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_759969_0while_lstm_cell_759971_0while_lstm_cell_759973_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759931┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_759969while_lstm_cell_759969_0"2
while_lstm_cell_759971while_lstm_cell_759971_0"2
while_lstm_cell_759973while_lstm_cell_759973_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name759973:&	"
 
_user_specified_name759971:&"
 
_user_specified_name759969:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬$
╬
while_body_760146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_760170_0:	 ђ+
while_lstm_cell_760172_0:	 ђ'
while_lstm_cell_760174_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_760170:	 ђ)
while_lstm_cell_760172:	 ђ%
while_lstm_cell_760174:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_760170_0while_lstm_cell_760172_0while_lstm_cell_760174_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760132┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_760170while_lstm_cell_760170_0"2
while_lstm_cell_760172while_lstm_cell_760172_0"2
while_lstm_cell_760174while_lstm_cell_760174_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name760174:&	"
 
_user_specified_name760172:&"
 
_user_specified_name760170:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ь
ќ
)__inference_dense_31_layer_call_fn_765319

inputs
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_761779o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765315:&"
 
_user_specified_name765313:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Х

e
F__inference_dropout_55_layer_call_and_return_conditional_losses_763368

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
┬	
├
while_cond_765052
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_765052___redundant_placeholder04
0while_while_cond_765052___redundant_placeholder14
0while_while_cond_765052___redundant_placeholder24
0while_while_cond_765052___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_35_layer_call_and_return_conditional_losses_763203

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763119*
condR
while_cond_763118*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ж8
х
while_body_761499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И8
щ
C__inference_lstm_36_layer_call_and_return_conditional_losses_760360

inputs#
lstm_cell_760278:	 ђ#
lstm_cell_760280:	 ђ
lstm_cell_760282:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_760278lstm_cell_760280lstm_cell_760282*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760277n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_760278lstm_cell_760280lstm_cell_760282*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_760291*
condR
while_cond_760290*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name760282:&"
 
_user_specified_name760280:&"
 
_user_specified_name760278:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
И8
щ
C__inference_lstm_35_layer_call_and_return_conditional_losses_759869

inputs#
lstm_cell_759787:	ђ#
lstm_cell_759789:	 ђ
lstm_cell_759791:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_759787lstm_cell_759789lstm_cell_759791*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759786n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_759787lstm_cell_759789lstm_cell_759791*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_759800*
condR
while_cond_759799*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name759791:&"
 
_user_specified_name759789:&"
 
_user_specified_name759787:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
┬	
├
while_cond_759944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_759944___redundant_placeholder04
0while_while_cond_759944___redundant_placeholder14
0while_while_cond_759944___redundant_placeholder24
0while_while_cond_759944___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╣
з
*__inference_lstm_cell_layer_call_fn_765589

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760478o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765581:&"
 
_user_specified_name765579:&"
 
_user_specified_name765577:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759931

inputs

states
states_11
matmul_readvariableop_resource:	ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ

e
F__inference_dropout_59_layer_call_and_return_conditional_losses_761796

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:          a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ИS
э
'sequential_11_lstm_37_while_body_759483H
Dsequential_11_lstm_37_while_sequential_11_lstm_37_while_loop_counterN
Jsequential_11_lstm_37_while_sequential_11_lstm_37_while_maximum_iterations+
'sequential_11_lstm_37_while_placeholder-
)sequential_11_lstm_37_while_placeholder_1-
)sequential_11_lstm_37_while_placeholder_2-
)sequential_11_lstm_37_while_placeholder_3G
Csequential_11_lstm_37_while_sequential_11_lstm_37_strided_slice_1_0Ѓ
sequential_11_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_37_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_11_lstm_37_while_lstm_cell_matmul_readvariableop_resource_0:	 ђ[
Hsequential_11_lstm_37_while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђV
Gsequential_11_lstm_37_while_lstm_cell_biasadd_readvariableop_resource_0:	ђ(
$sequential_11_lstm_37_while_identity*
&sequential_11_lstm_37_while_identity_1*
&sequential_11_lstm_37_while_identity_2*
&sequential_11_lstm_37_while_identity_3*
&sequential_11_lstm_37_while_identity_4*
&sequential_11_lstm_37_while_identity_5E
Asequential_11_lstm_37_while_sequential_11_lstm_37_strided_slice_1Ђ
}sequential_11_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_37_tensorarrayunstack_tensorlistfromtensorW
Dsequential_11_lstm_37_while_lstm_cell_matmul_readvariableop_resource:	 ђY
Fsequential_11_lstm_37_while_lstm_cell_matmul_1_readvariableop_resource:	 ђT
Esequential_11_lstm_37_while_lstm_cell_biasadd_readvariableop_resource:	ђѕб<sequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOpб;sequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOpб=sequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOpъ
Msequential_11/lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ћ
?sequential_11/lstm_37/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_37_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_37_while_placeholderVsequential_11/lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0├
;sequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpFsequential_11_lstm_37_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Ш
,sequential_11/lstm_37/while/lstm_cell/MatMulMatMulFsequential_11/lstm_37/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђК
=sequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHsequential_11_lstm_37_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0П
.sequential_11/lstm_37/while/lstm_cell/MatMul_1MatMul)sequential_11_lstm_37_while_placeholder_2Esequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђО
)sequential_11/lstm_37/while/lstm_cell/addAddV26sequential_11/lstm_37/while/lstm_cell/MatMul:product:08sequential_11/lstm_37/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ┴
<sequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGsequential_11_lstm_37_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0Я
-sequential_11/lstm_37/while/lstm_cell/BiasAddBiasAdd-sequential_11/lstm_37/while/lstm_cell/add:z:0Dsequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
5sequential_11/lstm_37/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :е
+sequential_11/lstm_37/while/lstm_cell/splitSplit>sequential_11/lstm_37/while/lstm_cell/split/split_dim:output:06sequential_11/lstm_37/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitа
-sequential_11/lstm_37/while/lstm_cell/SigmoidSigmoid4sequential_11/lstm_37/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_37/while/lstm_cell/Sigmoid_1Sigmoid4sequential_11/lstm_37/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ┬
)sequential_11/lstm_37/while/lstm_cell/mulMul3sequential_11/lstm_37/while/lstm_cell/Sigmoid_1:y:0)sequential_11_lstm_37_while_placeholder_3*
T0*'
_output_shapes
:          џ
*sequential_11/lstm_37/while/lstm_cell/ReluRelu4sequential_11/lstm_37/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Л
+sequential_11/lstm_37/while/lstm_cell/mul_1Mul1sequential_11/lstm_37/while/lstm_cell/Sigmoid:y:08sequential_11/lstm_37/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          к
+sequential_11/lstm_37/while/lstm_cell/add_1AddV2-sequential_11/lstm_37/while/lstm_cell/mul:z:0/sequential_11/lstm_37/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          б
/sequential_11/lstm_37/while/lstm_cell/Sigmoid_2Sigmoid4sequential_11/lstm_37/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:          Ќ
,sequential_11/lstm_37/while/lstm_cell/Relu_1Relu/sequential_11/lstm_37/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Н
+sequential_11/lstm_37/while/lstm_cell/mul_2Mul3sequential_11/lstm_37/while/lstm_cell/Sigmoid_2:y:0:sequential_11/lstm_37/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          џ
@sequential_11/lstm_37/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_37_while_placeholder_1'sequential_11_lstm_37_while_placeholder/sequential_11/lstm_37/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_11/lstm_37/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_11/lstm_37/while/addAddV2'sequential_11_lstm_37_while_placeholder*sequential_11/lstm_37/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_11/lstm_37/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_11/lstm_37/while/add_1AddV2Dsequential_11_lstm_37_while_sequential_11_lstm_37_while_loop_counter,sequential_11/lstm_37/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_11/lstm_37/while/IdentityIdentity%sequential_11/lstm_37/while/add_1:z:0!^sequential_11/lstm_37/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_11/lstm_37/while/Identity_1IdentityJsequential_11_lstm_37_while_sequential_11_lstm_37_while_maximum_iterations!^sequential_11/lstm_37/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_11/lstm_37/while/Identity_2Identity#sequential_11/lstm_37/while/add:z:0!^sequential_11/lstm_37/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_11/lstm_37/while/Identity_3IdentityPsequential_11/lstm_37/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_37/while/NoOp*
T0*
_output_shapes
: И
&sequential_11/lstm_37/while/Identity_4Identity/sequential_11/lstm_37/while/lstm_cell/mul_2:z:0!^sequential_11/lstm_37/while/NoOp*
T0*'
_output_shapes
:          И
&sequential_11/lstm_37/while/Identity_5Identity/sequential_11/lstm_37/while/lstm_cell/add_1:z:0!^sequential_11/lstm_37/while/NoOp*
T0*'
_output_shapes
:          ч
 sequential_11/lstm_37/while/NoOpNoOp=^sequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOp<^sequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOp>^sequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "Y
&sequential_11_lstm_37_while_identity_1/sequential_11/lstm_37/while/Identity_1:output:0"Y
&sequential_11_lstm_37_while_identity_2/sequential_11/lstm_37/while/Identity_2:output:0"Y
&sequential_11_lstm_37_while_identity_3/sequential_11/lstm_37/while/Identity_3:output:0"Y
&sequential_11_lstm_37_while_identity_4/sequential_11/lstm_37/while/Identity_4:output:0"Y
&sequential_11_lstm_37_while_identity_5/sequential_11/lstm_37/while/Identity_5:output:0"U
$sequential_11_lstm_37_while_identity-sequential_11/lstm_37/while/Identity:output:0"љ
Esequential_11_lstm_37_while_lstm_cell_biasadd_readvariableop_resourceGsequential_11_lstm_37_while_lstm_cell_biasadd_readvariableop_resource_0"њ
Fsequential_11_lstm_37_while_lstm_cell_matmul_1_readvariableop_resourceHsequential_11_lstm_37_while_lstm_cell_matmul_1_readvariableop_resource_0"ј
Dsequential_11_lstm_37_while_lstm_cell_matmul_readvariableop_resourceFsequential_11_lstm_37_while_lstm_cell_matmul_readvariableop_resource_0"ѕ
Asequential_11_lstm_37_while_sequential_11_lstm_37_strided_slice_1Csequential_11_lstm_37_while_sequential_11_lstm_37_strided_slice_1_0"ђ
}sequential_11_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_37_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_37_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2|
<sequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOp<sequential_11/lstm_37/while/lstm_cell/BiasAdd/ReadVariableOp2z
;sequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOp;sequential_11/lstm_37/while/lstm_cell/MatMul/ReadVariableOp2~
=sequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOp=sequential_11/lstm_37/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:uq

_output_shapes
: 
W
_user_specified_name?=sequential_11/lstm_37/TensorArrayUnstack/TensorListFromTensor:]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_37/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_37/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_37/while/loop_counter
§
и
(__inference_lstm_37_layer_call_fn_764027
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_760561|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764023:&"
 
_user_specified_name764021:&"
 
_user_specified_name764019:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
ж
d
F__inference_dropout_57_layer_call_and_return_conditional_losses_762283

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765442

inputs
states_0
states_11
matmul_readvariableop_resource:	ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ь
ч
'sequential_11_lstm_37_while_cond_759482H
Dsequential_11_lstm_37_while_sequential_11_lstm_37_while_loop_counterN
Jsequential_11_lstm_37_while_sequential_11_lstm_37_while_maximum_iterations+
'sequential_11_lstm_37_while_placeholder-
)sequential_11_lstm_37_while_placeholder_1-
)sequential_11_lstm_37_while_placeholder_2-
)sequential_11_lstm_37_while_placeholder_3J
Fsequential_11_lstm_37_while_less_sequential_11_lstm_37_strided_slice_1`
\sequential_11_lstm_37_while_sequential_11_lstm_37_while_cond_759482___redundant_placeholder0`
\sequential_11_lstm_37_while_sequential_11_lstm_37_while_cond_759482___redundant_placeholder1`
\sequential_11_lstm_37_while_sequential_11_lstm_37_while_cond_759482___redundant_placeholder2`
\sequential_11_lstm_37_while_sequential_11_lstm_37_while_cond_759482___redundant_placeholder3(
$sequential_11_lstm_37_while_identity
║
 sequential_11/lstm_37/while/LessLess'sequential_11_lstm_37_while_placeholderFsequential_11_lstm_37_while_less_sequential_11_lstm_37_strided_slice_1*
T0*
_output_shapes
: w
$sequential_11/lstm_37/while/IdentityIdentity$sequential_11/lstm_37/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_11_lstm_37_while_identity-sequential_11/lstm_37/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_37/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_37/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_37/while/loop_counter
▒
G
+__inference_dropout_56_layer_call_fn_763999

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_56_layer_call_and_return_conditional_losses_762127d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╩I
Ѕ
C__inference_lstm_37_layer_call_and_return_conditional_losses_762271

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_762187*
condR
while_cond_762186*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Х

e
F__inference_dropout_55_layer_call_and_return_conditional_losses_761276

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╩I
Ѕ
C__inference_lstm_35_layer_call_and_return_conditional_losses_763346

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763262*
condR
while_cond_763261*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ж8
х
while_body_763762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763989

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763905*
condR
while_cond_763904*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
э	
ш
D__inference_dense_32_layer_call_and_return_conditional_losses_765376

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
џ

e
F__inference_dropout_59_layer_call_and_return_conditional_losses_765352

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:          a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ж8
х
while_body_763619
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_36_layer_call_and_return_conditional_losses_762115

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_762031*
condR
while_cond_762030*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760132

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬$
╬
while_body_760637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_760661_0:	 ђ+
while_lstm_cell_760663_0:	 ђ'
while_lstm_cell_760665_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_760661:	 ђ)
while_lstm_cell_760663:	 ђ%
while_lstm_cell_760665:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_760661_0while_lstm_cell_760663_0while_lstm_cell_760665_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760623┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_760661while_lstm_cell_760661_0"2
while_lstm_cell_760663while_lstm_cell_760663_0"2
while_lstm_cell_760665while_lstm_cell_760665_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name760665:&	"
 
_user_specified_name760663:&"
 
_user_specified_name760661:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_763475
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763475___redundant_placeholder04
0while_while_cond_763475___redundant_placeholder14
0while_while_cond_763475___redundant_placeholder24
0while_while_cond_763475___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
§
и
(__inference_lstm_36_layer_call_fn_763384
inputs_0
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_760215|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name763380:&"
 
_user_specified_name763378:&"
 
_user_specified_name763376:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
┬	
├
while_cond_761874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_761874___redundant_placeholder04
0while_while_cond_761874___redundant_placeholder14
0while_while_cond_761874___redundant_placeholder24
0while_while_cond_761874___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_764907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764907___redundant_placeholder04
0while_while_cond_764907___redundant_placeholder14
0while_while_cond_764907___redundant_placeholder24
0while_while_cond_764907___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_761498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_761498___redundant_placeholder04
0while_while_cond_761498___redundant_placeholder14
0while_while_cond_761498___redundant_placeholder24
0while_while_cond_761498___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ѕJ
І
C__inference_lstm_36_layer_call_and_return_conditional_losses_763703
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_763619*
condR
while_cond_763618*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
э	
ш
D__inference_dense_32_layer_call_and_return_conditional_losses_761807

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765540

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
гђ
у$
"__inference__traced_restore_766285
file_prefix2
 assignvariableop_dense_31_kernel:  .
 assignvariableop_1_dense_31_bias: 4
"assignvariableop_2_dense_32_kernel: .
 assignvariableop_3_dense_32_bias:>
+assignvariableop_4_lstm_35_lstm_cell_kernel:	ђH
5assignvariableop_5_lstm_35_lstm_cell_recurrent_kernel:	 ђ8
)assignvariableop_6_lstm_35_lstm_cell_bias:	ђ>
+assignvariableop_7_lstm_36_lstm_cell_kernel:	 ђH
5assignvariableop_8_lstm_36_lstm_cell_recurrent_kernel:	 ђ8
)assignvariableop_9_lstm_36_lstm_cell_bias:	ђ?
,assignvariableop_10_lstm_37_lstm_cell_kernel:	 ђI
6assignvariableop_11_lstm_37_lstm_cell_recurrent_kernel:	 ђ9
*assignvariableop_12_lstm_37_lstm_cell_bias:	ђ?
,assignvariableop_13_lstm_38_lstm_cell_kernel:	 ђI
6assignvariableop_14_lstm_38_lstm_cell_recurrent_kernel:	 ђ9
*assignvariableop_15_lstm_38_lstm_cell_bias:	ђ'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: F
3assignvariableop_18_adam_m_lstm_35_lstm_cell_kernel:	ђF
3assignvariableop_19_adam_v_lstm_35_lstm_cell_kernel:	ђP
=assignvariableop_20_adam_m_lstm_35_lstm_cell_recurrent_kernel:	 ђP
=assignvariableop_21_adam_v_lstm_35_lstm_cell_recurrent_kernel:	 ђ@
1assignvariableop_22_adam_m_lstm_35_lstm_cell_bias:	ђ@
1assignvariableop_23_adam_v_lstm_35_lstm_cell_bias:	ђF
3assignvariableop_24_adam_m_lstm_36_lstm_cell_kernel:	 ђF
3assignvariableop_25_adam_v_lstm_36_lstm_cell_kernel:	 ђP
=assignvariableop_26_adam_m_lstm_36_lstm_cell_recurrent_kernel:	 ђP
=assignvariableop_27_adam_v_lstm_36_lstm_cell_recurrent_kernel:	 ђ@
1assignvariableop_28_adam_m_lstm_36_lstm_cell_bias:	ђ@
1assignvariableop_29_adam_v_lstm_36_lstm_cell_bias:	ђF
3assignvariableop_30_adam_m_lstm_37_lstm_cell_kernel:	 ђF
3assignvariableop_31_adam_v_lstm_37_lstm_cell_kernel:	 ђP
=assignvariableop_32_adam_m_lstm_37_lstm_cell_recurrent_kernel:	 ђP
=assignvariableop_33_adam_v_lstm_37_lstm_cell_recurrent_kernel:	 ђ@
1assignvariableop_34_adam_m_lstm_37_lstm_cell_bias:	ђ@
1assignvariableop_35_adam_v_lstm_37_lstm_cell_bias:	ђF
3assignvariableop_36_adam_m_lstm_38_lstm_cell_kernel:	 ђF
3assignvariableop_37_adam_v_lstm_38_lstm_cell_kernel:	 ђP
=assignvariableop_38_adam_m_lstm_38_lstm_cell_recurrent_kernel:	 ђP
=assignvariableop_39_adam_v_lstm_38_lstm_cell_recurrent_kernel:	 ђ@
1assignvariableop_40_adam_m_lstm_38_lstm_cell_bias:	ђ@
1assignvariableop_41_adam_v_lstm_38_lstm_cell_bias:	ђ<
*assignvariableop_42_adam_m_dense_31_kernel:  <
*assignvariableop_43_adam_v_dense_31_kernel:  6
(assignvariableop_44_adam_m_dense_31_bias: 6
(assignvariableop_45_adam_v_dense_31_bias: <
*assignvariableop_46_adam_m_dense_32_kernel: <
*assignvariableop_47_adam_v_dense_32_kernel: 6
(assignvariableop_48_adam_m_dense_32_bias:6
(assignvariableop_49_adam_v_dense_32_bias:%
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ђ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Д
valueЮBџ7B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▀
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ђ
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┤
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ы
_output_shapes▀
▄:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_32_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_32_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_4AssignVariableOp+assignvariableop_4_lstm_35_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_5AssignVariableOp5assignvariableop_5_lstm_35_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_6AssignVariableOp)assignvariableop_6_lstm_35_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_7AssignVariableOp+assignvariableop_7_lstm_36_lstm_cell_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_8AssignVariableOp5assignvariableop_8_lstm_36_lstm_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_9AssignVariableOp)assignvariableop_9_lstm_36_lstm_cell_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_10AssignVariableOp,assignvariableop_10_lstm_37_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_11AssignVariableOp6assignvariableop_11_lstm_37_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_12AssignVariableOp*assignvariableop_12_lstm_37_lstm_cell_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_13AssignVariableOp,assignvariableop_13_lstm_38_lstm_cell_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_14AssignVariableOp6assignvariableop_14_lstm_38_lstm_cell_recurrent_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_15AssignVariableOp*assignvariableop_15_lstm_38_lstm_cell_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_m_lstm_35_lstm_cell_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_v_lstm_35_lstm_cell_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_20AssignVariableOp=assignvariableop_20_adam_m_lstm_35_lstm_cell_recurrent_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_21AssignVariableOp=assignvariableop_21_adam_v_lstm_35_lstm_cell_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_m_lstm_35_lstm_cell_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_v_lstm_35_lstm_cell_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_m_lstm_36_lstm_cell_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_v_lstm_36_lstm_cell_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_m_lstm_36_lstm_cell_recurrent_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_v_lstm_36_lstm_cell_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_m_lstm_36_lstm_cell_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_v_lstm_36_lstm_cell_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_m_lstm_37_lstm_cell_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_v_lstm_37_lstm_cell_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_m_lstm_37_lstm_cell_recurrent_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_33AssignVariableOp=assignvariableop_33_adam_v_lstm_37_lstm_cell_recurrent_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_m_lstm_37_lstm_cell_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_v_lstm_37_lstm_cell_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_m_lstm_38_lstm_cell_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_v_lstm_38_lstm_cell_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_38AssignVariableOp=assignvariableop_38_adam_m_lstm_38_lstm_cell_recurrent_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_39AssignVariableOp=assignvariableop_39_adam_v_lstm_38_lstm_cell_recurrent_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_40AssignVariableOp1assignvariableop_40_adam_m_lstm_38_lstm_cell_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_v_lstm_38_lstm_cell_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_31_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_31_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_31_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_31_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_dense_32_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_dense_32_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_m_dense_32_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_v_dense_32_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 з	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: ╝	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*Ђ
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:'4#
!
_user_specified_name	count_1:'3#
!
_user_specified_name	total_1:420
.
_user_specified_nameAdam/v/dense_32/bias:410
.
_user_specified_nameAdam/m/dense_32/bias:602
0
_user_specified_nameAdam/v/dense_32/kernel:6/2
0
_user_specified_nameAdam/m/dense_32/kernel:4.0
.
_user_specified_nameAdam/v/dense_31/bias:4-0
.
_user_specified_nameAdam/m/dense_31/bias:6,2
0
_user_specified_nameAdam/v/dense_31/kernel:6+2
0
_user_specified_nameAdam/m/dense_31/kernel:=*9
7
_user_specified_nameAdam/v/lstm_38/lstm_cell/bias:=)9
7
_user_specified_nameAdam/m/lstm_38/lstm_cell/bias:I(E
C
_user_specified_name+)Adam/v/lstm_38/lstm_cell/recurrent_kernel:I'E
C
_user_specified_name+)Adam/m/lstm_38/lstm_cell/recurrent_kernel:?&;
9
_user_specified_name!Adam/v/lstm_38/lstm_cell/kernel:?%;
9
_user_specified_name!Adam/m/lstm_38/lstm_cell/kernel:=$9
7
_user_specified_nameAdam/v/lstm_37/lstm_cell/bias:=#9
7
_user_specified_nameAdam/m/lstm_37/lstm_cell/bias:I"E
C
_user_specified_name+)Adam/v/lstm_37/lstm_cell/recurrent_kernel:I!E
C
_user_specified_name+)Adam/m/lstm_37/lstm_cell/recurrent_kernel:? ;
9
_user_specified_name!Adam/v/lstm_37/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_37/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_36/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_36/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_36/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_36/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_36/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_36/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_35/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_35/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_35/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_35/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_35/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_35/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:62
0
_user_specified_namelstm_38/lstm_cell/bias:B>
<
_user_specified_name$"lstm_38/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_38/lstm_cell/kernel:62
0
_user_specified_namelstm_37/lstm_cell/bias:B>
<
_user_specified_name$"lstm_37/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_37/lstm_cell/kernel:6
2
0
_user_specified_namelstm_36/lstm_cell/bias:B	>
<
_user_specified_name$"lstm_36/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_36/lstm_cell/kernel:62
0
_user_specified_namelstm_35/lstm_cell/bias:B>
<
_user_specified_name$"lstm_35/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_35/lstm_cell/kernel:-)
'
_user_specified_namedense_32/bias:/+
)
_user_specified_namedense_32/kernel:-)
'
_user_specified_namedense_31/bias:/+
)
_user_specified_namedense_31/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤
d
+__inference_dropout_58_layer_call_fn_765288

inputs
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_58_layer_call_and_return_conditional_losses_761767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬	
├
while_cond_763618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763618___redundant_placeholder04
0while_while_cond_763618___redundant_placeholder14
0while_while_cond_763618___redundant_placeholder24
0while_while_cond_763618___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬<
ј
I__inference_sequential_11_layer_call_and_return_conditional_losses_761814
lstm_35_input!
lstm_35_761258:	ђ!
lstm_35_761260:	 ђ
lstm_35_761262:	ђ!
lstm_36_761421:	 ђ!
lstm_36_761423:	 ђ
lstm_36_761425:	ђ!
lstm_37_761584:	 ђ!
lstm_37_761586:	 ђ
lstm_37_761588:	ђ!
lstm_38_761749:	 ђ!
lstm_38_761751:	 ђ
lstm_38_761753:	ђ!
dense_31_761780:  
dense_31_761782: !
dense_32_761808: 
dense_32_761810:
identityѕб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallб"dropout_55/StatefulPartitionedCallб"dropout_56/StatefulPartitionedCallб"dropout_57/StatefulPartitionedCallб"dropout_58/StatefulPartitionedCallб"dropout_59/StatefulPartitionedCallбlstm_35/StatefulPartitionedCallбlstm_36/StatefulPartitionedCallбlstm_37/StatefulPartitionedCallбlstm_38/StatefulPartitionedCallЅ
lstm_35/StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputlstm_35_761258lstm_35_761260lstm_35_761262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_761257Ы
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_761276Д
lstm_36/StatefulPartitionedCallStatefulPartitionedCall+dropout_55/StatefulPartitionedCall:output:0lstm_36_761421lstm_36_761423lstm_36_761425*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_761420Ќ
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0#^dropout_55/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_56_layer_call_and_return_conditional_losses_761439Д
lstm_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_56/StatefulPartitionedCall:output:0lstm_37_761584lstm_37_761586lstm_37_761588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_761583Ќ
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall(lstm_37/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_57_layer_call_and_return_conditional_losses_761602Б
lstm_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_57/StatefulPartitionedCall:output:0lstm_38_761749lstm_38_761751lstm_38_761753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_38_layer_call_and_return_conditional_losses_761748Њ
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall(lstm_38/StatefulPartitionedCall:output:0#^dropout_57/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_58_layer_call_and_return_conditional_losses_761767Ћ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0dense_31_761780dense_31_761782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_761779ћ
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_761796Ћ
 dense_32/StatefulPartitionedCallStatefulPartitionedCall+dropout_59/StatefulPartitionedCall:output:0dense_32_761808dense_32_761810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_761807x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Е
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall ^lstm_38/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall2B
lstm_38/StatefulPartitionedCalllstm_38/StatefulPartitionedCall:&"
 
_user_specified_name761810:&"
 
_user_specified_name761808:&"
 
_user_specified_name761782:&"
 
_user_specified_name761780:&"
 
_user_specified_name761753:&"
 
_user_specified_name761751:&
"
 
_user_specified_name761749:&	"
 
_user_specified_name761588:&"
 
_user_specified_name761586:&"
 
_user_specified_name761584:&"
 
_user_specified_name761425:&"
 
_user_specified_name761423:&"
 
_user_specified_name761421:&"
 
_user_specified_name761262:&"
 
_user_specified_name761260:&"
 
_user_specified_name761258:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
┬	
├
while_cond_762186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_762186___redundant_placeholder04
0while_while_cond_762186___redundant_placeholder14
0while_while_cond_762186___redundant_placeholder24
0while_while_cond_762186___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ѕJ
І
C__inference_lstm_37_layer_call_and_return_conditional_losses_764346
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764262*
condR
while_cond_764261*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
ѕJ
І
C__inference_lstm_37_layer_call_and_return_conditional_losses_764203
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                   R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764119*
condR
while_cond_764118*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                   Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                   : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                   
"
_user_specified_name
inputs_0
є:
х
while_body_761663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_763761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_763761___redundant_placeholder04
0while_while_cond_763761___redundant_placeholder14
0while_while_cond_763761___redundant_placeholder24
0while_while_cond_763761___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
лJ
Ѕ
C__inference_lstm_38_layer_call_and_return_conditional_losses_761748

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_761663*
condR
while_cond_761662*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Ж8
х
while_body_764548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ж8
х
while_body_761336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765638

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
А
G
+__inference_dropout_59_layer_call_fn_765340

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_762452`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╦

ш
D__inference_dense_31_layer_call_and_return_conditional_losses_761779

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘
d
F__inference_dropout_59_layer_call_and_return_conditional_losses_765357

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ь
ч
'sequential_11_lstm_38_while_cond_759623H
Dsequential_11_lstm_38_while_sequential_11_lstm_38_while_loop_counterN
Jsequential_11_lstm_38_while_sequential_11_lstm_38_while_maximum_iterations+
'sequential_11_lstm_38_while_placeholder-
)sequential_11_lstm_38_while_placeholder_1-
)sequential_11_lstm_38_while_placeholder_2-
)sequential_11_lstm_38_while_placeholder_3J
Fsequential_11_lstm_38_while_less_sequential_11_lstm_38_strided_slice_1`
\sequential_11_lstm_38_while_sequential_11_lstm_38_while_cond_759623___redundant_placeholder0`
\sequential_11_lstm_38_while_sequential_11_lstm_38_while_cond_759623___redundant_placeholder1`
\sequential_11_lstm_38_while_sequential_11_lstm_38_while_cond_759623___redundant_placeholder2`
\sequential_11_lstm_38_while_sequential_11_lstm_38_while_cond_759623___redundant_placeholder3(
$sequential_11_lstm_38_while_identity
║
 sequential_11/lstm_38/while/LessLess'sequential_11_lstm_38_while_placeholderFsequential_11_lstm_38_while_less_sequential_11_lstm_38_strided_slice_1*
T0*
_output_shapes
: w
$sequential_11/lstm_38/while/IdentityIdentity$sequential_11/lstm_38/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_11_lstm_38_while_identity-sequential_11/lstm_38/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::]Y

_output_shapes
: 
?
_user_specified_name'%sequential_11/lstm_38/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_11/lstm_38/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_11/lstm_38/while/loop_counter
│
Ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765736

inputs
states_0
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ж8
х
while_body_763119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ж8
х
while_body_762187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ж
d
F__inference_dropout_55_layer_call_and_return_conditional_losses_763373

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
»
х
.__inference_sequential_11_layer_call_fn_762534
lstm_35_input
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
	unknown_2:	 ђ
	unknown_3:	 ђ
	unknown_4:	ђ
	unknown_5:	 ђ
	unknown_6:	 ђ
	unknown_7:	ђ
	unknown_8:	 ђ
	unknown_9:	 ђ

unknown_10:	ђ

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_762460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762530:&"
 
_user_specified_name762528:&"
 
_user_specified_name762526:&"
 
_user_specified_name762524:&"
 
_user_specified_name762522:&"
 
_user_specified_name762520:&
"
 
_user_specified_name762518:&	"
 
_user_specified_name762516:&"
 
_user_specified_name762514:&"
 
_user_specified_name762512:&"
 
_user_specified_name762510:&"
 
_user_specified_name762508:&"
 
_user_specified_name762506:&"
 
_user_specified_name762504:&"
 
_user_specified_name762502:&"
 
_user_specified_name762500:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
┘
d
F__inference_dropout_58_layer_call_and_return_conditional_losses_762441

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
є:
х
while_body_765053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ж8
х
while_body_764119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	 ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	 ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ж
d
F__inference_dropout_56_layer_call_and_return_conditional_losses_764016

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
 _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
я%
╬
while_body_760839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_760863_0:	 ђ+
while_lstm_cell_760865_0:	 ђ'
while_lstm_cell_760867_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_760863:	 ђ)
while_lstm_cell_760865:	 ђ%
while_lstm_cell_760867:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_760863_0while_lstm_cell_760865_0while_lstm_cell_760867_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760824r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_760863while_lstm_cell_760863_0"2
while_lstm_cell_760865while_lstm_cell_760865_0"2
while_lstm_cell_760867while_lstm_cell_760867_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name760867:&	"
 
_user_specified_name760865:&"
 
_user_specified_name760863:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
я%
╬
while_body_760986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_761010_0:	 ђ+
while_lstm_cell_761012_0:	 ђ'
while_lstm_cell_761014_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_761010:	 ђ)
while_lstm_cell_761012:	 ђ%
while_lstm_cell_761014:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_761010_0while_lstm_cell_761012_0while_lstm_cell_761014_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760971r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_761010while_lstm_cell_761010_0"2
while_lstm_cell_761012while_lstm_cell_761012_0"2
while_lstm_cell_761014while_lstm_cell_761014_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name761014:&	"
 
_user_specified_name761012:&"
 
_user_specified_name761010:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╩I
Ѕ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764632

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764548*
condR
while_cond_764547*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
А
G
+__inference_dropout_58_layer_call_fn_765293

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_58_layer_call_and_return_conditional_losses_762441`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
М
х
(__inference_lstm_35_layer_call_fn_762763

inputs
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_761257s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762759:&"
 
_user_specified_name762757:&"
 
_user_specified_name762755:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ж8
х
while_body_762833
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_760838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760838___redundant_placeholder04
0while_while_cond_760838___redundant_placeholder14
0while_while_cond_760838___redundant_placeholder24
0while_while_cond_760838___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_760145
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_760145___redundant_placeholder04
0while_while_cond_760145___redundant_placeholder14
0while_while_cond_760145___redundant_placeholder24
0while_while_cond_760145___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
§
Ф
$__inference_signature_wrapper_762730
lstm_35_input
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
	unknown_2:	 ђ
	unknown_3:	 ђ
	unknown_4:	ђ
	unknown_5:	 ђ
	unknown_6:	 ђ
	unknown_7:	ђ
	unknown_8:	 ђ
	unknown_9:	 ђ

unknown_10:	ђ

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_759724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         
: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name762726:&"
 
_user_specified_name762724:&"
 
_user_specified_name762722:&"
 
_user_specified_name762720:&"
 
_user_specified_name762718:&"
 
_user_specified_name762716:&
"
 
_user_specified_name762714:&	"
 
_user_specified_name762712:&"
 
_user_specified_name762710:&"
 
_user_specified_name762708:&"
 
_user_specified_name762706:&"
 
_user_specified_name762704:&"
 
_user_specified_name762702:&"
 
_user_specified_name762700:&"
 
_user_specified_name762698:&"
 
_user_specified_name762696:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_35_input
Ж8
х
while_body_761173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▀
d
+__inference_dropout_55_layer_call_fn_763351

inputs
identityѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_761276s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╩I
Ѕ
C__inference_lstm_35_layer_call_and_return_conditional_losses_761257

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_761173*
condR
while_cond_761172*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╣
з
*__inference_lstm_cell_layer_call_fn_765606

inputs
states_0
states_1
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765598:&"
 
_user_specified_name765596:&"
 
_user_specified_name765594:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
лJ
Ѕ
C__inference_lstm_38_layer_call_and_return_conditional_losses_765283

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_765198*
condR
while_cond_765197*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╩I
Ѕ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764489

inputs;
(lstm_cell_matmul_readvariableop_resource:	 ђ=
*lstm_cell_matmul_1_readvariableop_resource:	 ђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
          R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЇ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:          q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:          }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:          _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▄
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_764405*
condR
while_cond_764404*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         
 Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
▒
G
+__inference_dropout_55_layer_call_fn_763356

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_761971d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
М
х
(__inference_lstm_37_layer_call_fn_764060

inputs
unknown:	 ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_37_layer_call_and_return_conditional_losses_762271s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
 <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name764056:&"
 
_user_specified_name764054:&"
 
_user_specified_name764052:S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
╣
з
*__inference_lstm_cell_layer_call_fn_765410

inputs
states_0
states_1
unknown:	ђ
	unknown_0:	 ђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_759931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name765402:&"
 
_user_specified_name765400:&"
 
_user_specified_name765398:QM
'
_output_shapes
:          
"
_user_specified_name
states_1:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ж8
х
while_body_762976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђE
2while_lstm_cell_matmul_1_readvariableop_resource_0:	 ђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђC
0while_lstm_cell_matmul_1_readvariableop_resource:	 ђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:          ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:          Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:          ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:          v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:          k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:          Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:          ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          Б

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬$
╬
while_body_760492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_760516_0:	 ђ+
while_lstm_cell_760518_0:	 ђ'
while_lstm_cell_760520_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_760516:	 ђ)
while_lstm_cell_760518:	 ђ%
while_lstm_cell_760520:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_760516_0while_lstm_cell_760518_0while_lstm_cell_760520_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760478┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_760516while_lstm_cell_760516_0"2
while_lstm_cell_760518while_lstm_cell_760518_0"2
while_lstm_cell_760520while_lstm_cell_760520_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name760520:&	"
 
_user_specified_name760518:&"
 
_user_specified_name760516:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬	
├
while_cond_764762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764762___redundant_placeholder04
0while_while_cond_764762___redundant_placeholder14
0while_while_cond_764762___redundant_placeholder24
0while_while_cond_764762___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760824

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ф
Ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_760478

inputs

states
states_11
matmul_readvariableop_resource:	 ђ3
 matmul_1_readvariableop_resource:	 ђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:          :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Х

e
F__inference_dropout_56_layer_call_and_return_conditional_losses_761439

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs
Х

e
F__inference_dropout_57_layer_call_and_return_conditional_losses_764654

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
 Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
 T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ќ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
 e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
 :S O
+
_output_shapes
:         
 
 
_user_specified_nameinputs"ДL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultД
K
lstm_35_input:
serving_default_lstm_35_input:0         
<
dense_320
StatefulPartitionedCall:0         tensorflow/serving/predict:оЇ
њ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╝
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_random_generator"
_tf_keras_layer
┌
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator
,cell
-
state_spec"
_tf_keras_rnn_layer
╝
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator"
_tf_keras_layer
┌
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator
<cell
=
state_spec"
_tf_keras_rnn_layer
╝
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
D_random_generator"
_tf_keras_layer
┌
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator
Lcell
M
state_spec"
_tf_keras_rnn_layer
╝
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator"
_tf_keras_layer
╗
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
╝
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
c_random_generator"
_tf_keras_layer
╗
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias"
_tf_keras_layer
ќ
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11
[12
\13
j14
k15"
trackable_list_wrapper
ќ
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11
[12
\13
j14
k15"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¤
}trace_0
~trace_12ў
.__inference_sequential_11_layer_call_fn_762497
.__inference_sequential_11_layer_call_fn_762534х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z}trace_0z~trace_1
Є
trace_0
ђtrace_12╬
I__inference_sequential_11_layer_call_and_return_conditional_losses_761814
I__inference_sequential_11_layer_call_and_return_conditional_losses_762460х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0zђtrace_1
мB¤
!__inference__wrapped_model_759724lstm_35_input"ў
Љ▓Ї
FullArgSpec
argsџ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Б
Ђ
_variables
ѓ_iterations
Ѓ_learning_rate
ё_index_dict
Ё
_momentums
є_velocities
Є_update_step_xla"
experimentalOptimizer
-
ѕserving_default"
signature_map
5
l0
m1
n2"
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
Ѕstates
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
У
Јtrace_0
љtrace_1
Љtrace_2
њtrace_32ш
(__inference_lstm_35_layer_call_fn_762741
(__inference_lstm_35_layer_call_fn_762752
(__inference_lstm_35_layer_call_fn_762763
(__inference_lstm_35_layer_call_fn_762774╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0zљtrace_1zЉtrace_2zњtrace_3
н
Њtrace_0
ћtrace_1
Ћtrace_2
ќtrace_32р
C__inference_lstm_35_layer_call_and_return_conditional_losses_762917
C__inference_lstm_35_layer_call_and_return_conditional_losses_763060
C__inference_lstm_35_layer_call_and_return_conditional_losses_763203
C__inference_lstm_35_layer_call_and_return_conditional_losses_763346╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0zћtrace_1zЋtrace_2zќtrace_3
"
_generic_user_object
ђ
Ќ	variables
ўtrainable_variables
Ўregularization_losses
џ	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses
Ю_random_generator
ъ
state_size

lkernel
mrecurrent_kernel
nbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
┴
цtrace_0
Цtrace_12є
+__inference_dropout_55_layer_call_fn_763351
+__inference_dropout_55_layer_call_fn_763356Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0zЦtrace_1
э
дtrace_0
Дtrace_12╝
F__inference_dropout_55_layer_call_and_return_conditional_losses_763368
F__inference_dropout_55_layer_call_and_return_conditional_losses_763373Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zдtrace_0zДtrace_1
"
_generic_user_object
5
o0
p1
q2"
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
еstates
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
У
«trace_0
»trace_1
░trace_2
▒trace_32ш
(__inference_lstm_36_layer_call_fn_763384
(__inference_lstm_36_layer_call_fn_763395
(__inference_lstm_36_layer_call_fn_763406
(__inference_lstm_36_layer_call_fn_763417╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0z»trace_1z░trace_2z▒trace_3
н
▓trace_0
│trace_1
┤trace_2
хtrace_32р
C__inference_lstm_36_layer_call_and_return_conditional_losses_763560
C__inference_lstm_36_layer_call_and_return_conditional_losses_763703
C__inference_lstm_36_layer_call_and_return_conditional_losses_763846
C__inference_lstm_36_layer_call_and_return_conditional_losses_763989╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▓trace_0z│trace_1z┤trace_2zхtrace_3
"
_generic_user_object
ђ
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses
╝_random_generator
й
state_size

okernel
precurrent_kernel
qbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
┴
├trace_0
─trace_12є
+__inference_dropout_56_layer_call_fn_763994
+__inference_dropout_56_layer_call_fn_763999Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├trace_0z─trace_1
э
┼trace_0
кtrace_12╝
F__inference_dropout_56_layer_call_and_return_conditional_losses_764011
F__inference_dropout_56_layer_call_and_return_conditional_losses_764016Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼trace_0zкtrace_1
"
_generic_user_object
5
r0
s1
t2"
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
Кstates
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
У
═trace_0
╬trace_1
¤trace_2
лtrace_32ш
(__inference_lstm_37_layer_call_fn_764027
(__inference_lstm_37_layer_call_fn_764038
(__inference_lstm_37_layer_call_fn_764049
(__inference_lstm_37_layer_call_fn_764060╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z═trace_0z╬trace_1z¤trace_2zлtrace_3
н
Лtrace_0
мtrace_1
Мtrace_2
нtrace_32р
C__inference_lstm_37_layer_call_and_return_conditional_losses_764203
C__inference_lstm_37_layer_call_and_return_conditional_losses_764346
C__inference_lstm_37_layer_call_and_return_conditional_losses_764489
C__inference_lstm_37_layer_call_and_return_conditional_losses_764632╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЛtrace_0zмtrace_1zМtrace_2zнtrace_3
"
_generic_user_object
ђ
Н	variables
оtrainable_variables
Оregularization_losses
п	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses
█_random_generator
▄
state_size

rkernel
srecurrent_kernel
tbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
┴
Рtrace_0
сtrace_12є
+__inference_dropout_57_layer_call_fn_764637
+__inference_dropout_57_layer_call_fn_764642Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zРtrace_0zсtrace_1
э
Сtrace_0
тtrace_12╝
F__inference_dropout_57_layer_call_and_return_conditional_losses_764654
F__inference_dropout_57_layer_call_and_return_conditional_losses_764659Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0zтtrace_1
"
_generic_user_object
5
u0
v1
w2"
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
Тstates
уnon_trainable_variables
Уlayers
жmetrics
 Жlayer_regularization_losses
вlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
У
Вtrace_0
ьtrace_1
Ьtrace_2
№trace_32ш
(__inference_lstm_38_layer_call_fn_764670
(__inference_lstm_38_layer_call_fn_764681
(__inference_lstm_38_layer_call_fn_764692
(__inference_lstm_38_layer_call_fn_764703╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zВtrace_0zьtrace_1zЬtrace_2z№trace_3
н
­trace_0
ыtrace_1
Ыtrace_2
зtrace_32р
C__inference_lstm_38_layer_call_and_return_conditional_losses_764848
C__inference_lstm_38_layer_call_and_return_conditional_losses_764993
C__inference_lstm_38_layer_call_and_return_conditional_losses_765138
C__inference_lstm_38_layer_call_and_return_conditional_losses_765283╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0zыtrace_1zЫtrace_2zзtrace_3
"
_generic_user_object
ђ
З	variables
шtrainable_variables
Шregularization_losses
э	keras_api
Э__call__
+щ&call_and_return_all_conditional_losses
Щ_random_generator
ч
state_size

ukernel
vrecurrent_kernel
wbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
┴
Ђtrace_0
ѓtrace_12є
+__inference_dropout_58_layer_call_fn_765288
+__inference_dropout_58_layer_call_fn_765293Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0zѓtrace_1
э
Ѓtrace_0
ёtrace_12╝
F__inference_dropout_58_layer_call_and_return_conditional_losses_765305
F__inference_dropout_58_layer_call_and_return_conditional_losses_765310Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЃtrace_0zёtrace_1
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
т
іtrace_02к
)__inference_dense_31_layer_call_fn_765319ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zіtrace_0
ђ
Іtrace_02р
D__inference_dense_31_layer_call_and_return_conditional_losses_765330ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІtrace_0
!:  2dense_31/kernel
: 2dense_31/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
┴
Љtrace_0
њtrace_12є
+__inference_dropout_59_layer_call_fn_765335
+__inference_dropout_59_layer_call_fn_765340Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0zњtrace_1
э
Њtrace_0
ћtrace_12╝
F__inference_dropout_59_layer_call_and_return_conditional_losses_765352
F__inference_dropout_59_layer_call_and_return_conditional_losses_765357Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0zћtrace_1
"
_generic_user_object
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
т
џtrace_02к
)__inference_dense_32_layer_call_fn_765366ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0
ђ
Џtrace_02р
D__inference_dense_32_layer_call_and_return_conditional_losses_765376ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЏtrace_0
!: 2dense_32/kernel
:2dense_32/bias
+:)	ђ2lstm_35/lstm_cell/kernel
5:3	 ђ2"lstm_35/lstm_cell/recurrent_kernel
%:#ђ2lstm_35/lstm_cell/bias
+:)	 ђ2lstm_36/lstm_cell/kernel
5:3	 ђ2"lstm_36/lstm_cell/recurrent_kernel
%:#ђ2lstm_36/lstm_cell/bias
+:)	 ђ2lstm_37/lstm_cell/kernel
5:3	 ђ2"lstm_37/lstm_cell/recurrent_kernel
%:#ђ2lstm_37/lstm_cell/bias
+:)	 ђ2lstm_38/lstm_cell/kernel
5:3	 ђ2"lstm_38/lstm_cell/recurrent_kernel
%:#ђ2lstm_38/lstm_cell/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
ю0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зB­
.__inference_sequential_11_layer_call_fn_762497lstm_35_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
зB­
.__inference_sequential_11_layer_call_fn_762534lstm_35_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
I__inference_sequential_11_layer_call_and_return_conditional_losses_761814lstm_35_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
I__inference_sequential_11_layer_call_and_return_conditional_losses_762460lstm_35_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┐
ѓ0
ъ1
Ъ2
а3
А4
б5
Б6
ц7
Ц8
д9
Д10
е11
Е12
ф13
Ф14
г15
Г16
«17
»18
░19
▒20
▓21
│22
┤23
х24
Х25
и26
И27
╣28
║29
╗30
╝31
й32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
д
ъ0
а1
б2
ц3
д4
е5
ф6
г7
«8
░9
▓10
┤11
Х12
И13
║14
╝15"
trackable_list_wrapper
д
Ъ0
А1
Б2
Ц3
Д4
Е5
Ф6
Г7
»8
▒9
│10
х11
и12
╣13
╗14
й15"
trackable_list_wrapper
х2▓»
д▓б
FullArgSpec*
args"џ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
▄B┘
$__inference_signature_wrapper_762730lstm_35_input"Ъ
ў▓ћ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 "

kwonlyargsџ
jlstm_35_input
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_35_layer_call_fn_762741inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_lstm_35_layer_call_fn_762752inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_35_layer_call_fn_762763inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_35_layer_call_fn_762774inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_35_layer_call_and_return_conditional_losses_762917inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_35_layer_call_and_return_conditional_losses_763060inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_35_layer_call_and_return_conditional_losses_763203inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_35_layer_call_and_return_conditional_losses_763346inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
l0
m1
n2"
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
Ќ	variables
ўtrainable_variables
Ўregularization_losses
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
╔
├trace_0
─trace_12ј
*__inference_lstm_cell_layer_call_fn_765393
*__inference_lstm_cell_layer_call_fn_765410│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├trace_0z─trace_1
 
┼trace_0
кtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765442
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765474│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼trace_0zкtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
+__inference_dropout_55_layer_call_fn_763351inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
рBя
+__inference_dropout_55_layer_call_fn_763356inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_55_layer_call_and_return_conditional_losses_763368inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_55_layer_call_and_return_conditional_losses_763373inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_36_layer_call_fn_763384inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_lstm_36_layer_call_fn_763395inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_36_layer_call_fn_763406inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_36_layer_call_fn_763417inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763560inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763703inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763846inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_36_layer_call_and_return_conditional_losses_763989inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
o0
p1
q2"
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Х	variables
иtrainable_variables
Иregularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
╔
╠trace_0
═trace_12ј
*__inference_lstm_cell_layer_call_fn_765491
*__inference_lstm_cell_layer_call_fn_765508│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠trace_0z═trace_1
 
╬trace_0
¤trace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765540
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765572│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╬trace_0z¤trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
+__inference_dropout_56_layer_call_fn_763994inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
рBя
+__inference_dropout_56_layer_call_fn_763999inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_56_layer_call_and_return_conditional_losses_764011inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_56_layer_call_and_return_conditional_losses_764016inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_37_layer_call_fn_764027inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_lstm_37_layer_call_fn_764038inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_37_layer_call_fn_764049inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_37_layer_call_fn_764060inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764203inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764346inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764489inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_37_layer_call_and_return_conditional_losses_764632inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
r0
s1
t2"
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
Н	variables
оtrainable_variables
Оregularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
╔
Нtrace_0
оtrace_12ј
*__inference_lstm_cell_layer_call_fn_765589
*__inference_lstm_cell_layer_call_fn_765606│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zНtrace_0zоtrace_1
 
Оtrace_0
пtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765638
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765670│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zОtrace_0zпtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
+__inference_dropout_57_layer_call_fn_764637inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
рBя
+__inference_dropout_57_layer_call_fn_764642inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_57_layer_call_and_return_conditional_losses_764654inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_57_layer_call_and_return_conditional_losses_764659inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_38_layer_call_fn_764670inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_lstm_38_layer_call_fn_764681inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_38_layer_call_fn_764692inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
(__inference_lstm_38_layer_call_fn_764703inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_38_layer_call_and_return_conditional_losses_764848inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_lstm_38_layer_call_and_return_conditional_losses_764993inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_38_layer_call_and_return_conditional_losses_765138inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
C__inference_lstm_38_layer_call_and_return_conditional_losses_765283inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
u0
v1
w2"
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
Пlayer_metrics
З	variables
шtrainable_variables
Шregularization_losses
Э__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
╔
яtrace_0
▀trace_12ј
*__inference_lstm_cell_layer_call_fn_765687
*__inference_lstm_cell_layer_call_fn_765704│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0z▀trace_1
 
Яtrace_0
рtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765736
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765768│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0zрtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
+__inference_dropout_58_layer_call_fn_765288inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
рBя
+__inference_dropout_58_layer_call_fn_765293inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_58_layer_call_and_return_conditional_losses_765305inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_58_layer_call_and_return_conditional_losses_765310inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBл
)__inference_dense_31_layer_call_fn_765319inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЬBв
D__inference_dense_31_layer_call_and_return_conditional_losses_765330inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
+__inference_dropout_59_layer_call_fn_765335inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
рBя
+__inference_dropout_59_layer_call_fn_765340inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_59_layer_call_and_return_conditional_losses_765352inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
F__inference_dropout_59_layer_call_and_return_conditional_losses_765357inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBл
)__inference_dense_32_layer_call_fn_765366inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЬBв
D__inference_dense_32_layer_call_and_return_conditional_losses_765376inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
Р	variables
с	keras_api

Сtotal

тcount"
_tf_keras_metric
c
Т	variables
у	keras_api

Уtotal

жcount
Ж
_fn_kwargs"
_tf_keras_metric
0:.	ђ2Adam/m/lstm_35/lstm_cell/kernel
0:.	ђ2Adam/v/lstm_35/lstm_cell/kernel
::8	 ђ2)Adam/m/lstm_35/lstm_cell/recurrent_kernel
::8	 ђ2)Adam/v/lstm_35/lstm_cell/recurrent_kernel
*:(ђ2Adam/m/lstm_35/lstm_cell/bias
*:(ђ2Adam/v/lstm_35/lstm_cell/bias
0:.	 ђ2Adam/m/lstm_36/lstm_cell/kernel
0:.	 ђ2Adam/v/lstm_36/lstm_cell/kernel
::8	 ђ2)Adam/m/lstm_36/lstm_cell/recurrent_kernel
::8	 ђ2)Adam/v/lstm_36/lstm_cell/recurrent_kernel
*:(ђ2Adam/m/lstm_36/lstm_cell/bias
*:(ђ2Adam/v/lstm_36/lstm_cell/bias
0:.	 ђ2Adam/m/lstm_37/lstm_cell/kernel
0:.	 ђ2Adam/v/lstm_37/lstm_cell/kernel
::8	 ђ2)Adam/m/lstm_37/lstm_cell/recurrent_kernel
::8	 ђ2)Adam/v/lstm_37/lstm_cell/recurrent_kernel
*:(ђ2Adam/m/lstm_37/lstm_cell/bias
*:(ђ2Adam/v/lstm_37/lstm_cell/bias
0:.	 ђ2Adam/m/lstm_38/lstm_cell/kernel
0:.	 ђ2Adam/v/lstm_38/lstm_cell/kernel
::8	 ђ2)Adam/m/lstm_38/lstm_cell/recurrent_kernel
::8	 ђ2)Adam/v/lstm_38/lstm_cell/recurrent_kernel
*:(ђ2Adam/m/lstm_38/lstm_cell/bias
*:(ђ2Adam/v/lstm_38/lstm_cell/bias
&:$  2Adam/m/dense_31/kernel
&:$  2Adam/v/dense_31/kernel
 : 2Adam/m/dense_31/bias
 : 2Adam/v/dense_31/bias
&:$ 2Adam/m/dense_32/kernel
&:$ 2Adam/v/dense_32/kernel
 :2Adam/m/dense_32/bias
 :2Adam/v/dense_32/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■Bч
*__inference_lstm_cell_layer_call_fn_765393inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
*__inference_lstm_cell_layer_call_fn_765410inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765442inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765474inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■Bч
*__inference_lstm_cell_layer_call_fn_765491inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
*__inference_lstm_cell_layer_call_fn_765508inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765540inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765572inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■Bч
*__inference_lstm_cell_layer_call_fn_765589inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
*__inference_lstm_cell_layer_call_fn_765606inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765638inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765670inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■Bч
*__inference_lstm_cell_layer_call_fn_765687inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
*__inference_lstm_cell_layer_call_fn_765704inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765736inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765768inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
С0
т1"
trackable_list_wrapper
.
Р	variables"
_generic_user_object
:  (2total
:  (2count
0
У0
ж1"
trackable_list_wrapper
.
Т	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЕ
!__inference__wrapped_model_759724Ѓlmnopqrstuvw[\jk:б7
0б-
+і(
lstm_35_input         

ф "3ф0
.
dense_32"і
dense_32         Ф
D__inference_dense_31_layer_call_and_return_conditional_losses_765330c[\/б,
%б"
 і
inputs          
ф ",б)
"і
tensor_0          
џ Ё
)__inference_dense_31_layer_call_fn_765319X[\/б,
%б"
 і
inputs          
ф "!і
unknown          Ф
D__inference_dense_32_layer_call_and_return_conditional_losses_765376cjk/б,
%б"
 і
inputs          
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_32_layer_call_fn_765366Xjk/б,
%б"
 і
inputs          
ф "!і
unknown         х
F__inference_dropout_55_layer_call_and_return_conditional_losses_763368k7б4
-б*
$і!
inputs         
 
p
ф "0б-
&і#
tensor_0         
 
џ х
F__inference_dropout_55_layer_call_and_return_conditional_losses_763373k7б4
-б*
$і!
inputs         
 
p 
ф "0б-
&і#
tensor_0         
 
џ Ј
+__inference_dropout_55_layer_call_fn_763351`7б4
-б*
$і!
inputs         
 
p
ф "%і"
unknown         
 Ј
+__inference_dropout_55_layer_call_fn_763356`7б4
-б*
$і!
inputs         
 
p 
ф "%і"
unknown         
 х
F__inference_dropout_56_layer_call_and_return_conditional_losses_764011k7б4
-б*
$і!
inputs         
 
p
ф "0б-
&і#
tensor_0         
 
џ х
F__inference_dropout_56_layer_call_and_return_conditional_losses_764016k7б4
-б*
$і!
inputs         
 
p 
ф "0б-
&і#
tensor_0         
 
џ Ј
+__inference_dropout_56_layer_call_fn_763994`7б4
-б*
$і!
inputs         
 
p
ф "%і"
unknown         
 Ј
+__inference_dropout_56_layer_call_fn_763999`7б4
-б*
$і!
inputs         
 
p 
ф "%і"
unknown         
 х
F__inference_dropout_57_layer_call_and_return_conditional_losses_764654k7б4
-б*
$і!
inputs         
 
p
ф "0б-
&і#
tensor_0         
 
џ х
F__inference_dropout_57_layer_call_and_return_conditional_losses_764659k7б4
-б*
$і!
inputs         
 
p 
ф "0б-
&і#
tensor_0         
 
џ Ј
+__inference_dropout_57_layer_call_fn_764637`7б4
-б*
$і!
inputs         
 
p
ф "%і"
unknown         
 Ј
+__inference_dropout_57_layer_call_fn_764642`7б4
-б*
$і!
inputs         
 
p 
ф "%і"
unknown         
 Г
F__inference_dropout_58_layer_call_and_return_conditional_losses_765305c3б0
)б&
 і
inputs          
p
ф ",б)
"і
tensor_0          
џ Г
F__inference_dropout_58_layer_call_and_return_conditional_losses_765310c3б0
)б&
 і
inputs          
p 
ф ",б)
"і
tensor_0          
џ Є
+__inference_dropout_58_layer_call_fn_765288X3б0
)б&
 і
inputs          
p
ф "!і
unknown          Є
+__inference_dropout_58_layer_call_fn_765293X3б0
)б&
 і
inputs          
p 
ф "!і
unknown          Г
F__inference_dropout_59_layer_call_and_return_conditional_losses_765352c3б0
)б&
 і
inputs          
p
ф ",б)
"і
tensor_0          
џ Г
F__inference_dropout_59_layer_call_and_return_conditional_losses_765357c3б0
)б&
 і
inputs          
p 
ф ",б)
"і
tensor_0          
џ Є
+__inference_dropout_59_layer_call_fn_765335X3б0
)б&
 і
inputs          
p
ф "!і
unknown          Є
+__inference_dropout_59_layer_call_fn_765340X3б0
)б&
 і
inputs          
p 
ф "!і
unknown          ┘
C__inference_lstm_35_layer_call_and_return_conditional_losses_762917ЉlmnOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "9б6
/і,
tensor_0                   
џ ┘
C__inference_lstm_35_layer_call_and_return_conditional_losses_763060ЉlmnOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "9б6
/і,
tensor_0                   
џ ┐
C__inference_lstm_35_layer_call_and_return_conditional_losses_763203xlmn?б<
5б2
$і!
inputs         


 
p

 
ф "0б-
&і#
tensor_0         
 
џ ┐
C__inference_lstm_35_layer_call_and_return_conditional_losses_763346xlmn?б<
5б2
$і!
inputs         


 
p 

 
ф "0б-
&і#
tensor_0         
 
џ │
(__inference_lstm_35_layer_call_fn_762741єlmnOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ".і+
unknown                   │
(__inference_lstm_35_layer_call_fn_762752єlmnOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ".і+
unknown                   Ў
(__inference_lstm_35_layer_call_fn_762763mlmn?б<
5б2
$і!
inputs         


 
p

 
ф "%і"
unknown         
 Ў
(__inference_lstm_35_layer_call_fn_762774mlmn?б<
5б2
$і!
inputs         


 
p 

 
ф "%і"
unknown         
 ┘
C__inference_lstm_36_layer_call_and_return_conditional_losses_763560ЉopqOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф "9б6
/і,
tensor_0                   
џ ┘
C__inference_lstm_36_layer_call_and_return_conditional_losses_763703ЉopqOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф "9б6
/і,
tensor_0                   
џ ┐
C__inference_lstm_36_layer_call_and_return_conditional_losses_763846xopq?б<
5б2
$і!
inputs         
 

 
p

 
ф "0б-
&і#
tensor_0         
 
џ ┐
C__inference_lstm_36_layer_call_and_return_conditional_losses_763989xopq?б<
5б2
$і!
inputs         
 

 
p 

 
ф "0б-
&і#
tensor_0         
 
џ │
(__inference_lstm_36_layer_call_fn_763384єopqOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф ".і+
unknown                   │
(__inference_lstm_36_layer_call_fn_763395єopqOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф ".і+
unknown                   Ў
(__inference_lstm_36_layer_call_fn_763406mopq?б<
5б2
$і!
inputs         
 

 
p

 
ф "%і"
unknown         
 Ў
(__inference_lstm_36_layer_call_fn_763417mopq?б<
5б2
$і!
inputs         
 

 
p 

 
ф "%і"
unknown         
 ┘
C__inference_lstm_37_layer_call_and_return_conditional_losses_764203ЉrstOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф "9б6
/і,
tensor_0                   
џ ┘
C__inference_lstm_37_layer_call_and_return_conditional_losses_764346ЉrstOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф "9б6
/і,
tensor_0                   
џ ┐
C__inference_lstm_37_layer_call_and_return_conditional_losses_764489xrst?б<
5б2
$і!
inputs         
 

 
p

 
ф "0б-
&і#
tensor_0         
 
џ ┐
C__inference_lstm_37_layer_call_and_return_conditional_losses_764632xrst?б<
5б2
$і!
inputs         
 

 
p 

 
ф "0б-
&і#
tensor_0         
 
џ │
(__inference_lstm_37_layer_call_fn_764027єrstOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф ".і+
unknown                   │
(__inference_lstm_37_layer_call_fn_764038єrstOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф ".і+
unknown                   Ў
(__inference_lstm_37_layer_call_fn_764049mrst?б<
5б2
$і!
inputs         
 

 
p

 
ф "%і"
unknown         
 Ў
(__inference_lstm_37_layer_call_fn_764060mrst?б<
5б2
$і!
inputs         
 

 
p 

 
ф "%і"
unknown         
 ╠
C__inference_lstm_38_layer_call_and_return_conditional_losses_764848ёuvwOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф ",б)
"і
tensor_0          
џ ╠
C__inference_lstm_38_layer_call_and_return_conditional_losses_764993ёuvwOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф ",б)
"і
tensor_0          
џ ╗
C__inference_lstm_38_layer_call_and_return_conditional_losses_765138tuvw?б<
5б2
$і!
inputs         
 

 
p

 
ф ",б)
"і
tensor_0          
џ ╗
C__inference_lstm_38_layer_call_and_return_conditional_losses_765283tuvw?б<
5б2
$і!
inputs         
 

 
p 

 
ф ",б)
"і
tensor_0          
џ Ц
(__inference_lstm_38_layer_call_fn_764670yuvwOбL
EбB
4џ1
/і,
inputs_0                   

 
p

 
ф "!і
unknown          Ц
(__inference_lstm_38_layer_call_fn_764681yuvwOбL
EбB
4џ1
/і,
inputs_0                   

 
p 

 
ф "!і
unknown          Ћ
(__inference_lstm_38_layer_call_fn_764692iuvw?б<
5б2
$і!
inputs         
 

 
p

 
ф "!і
unknown          Ћ
(__inference_lstm_38_layer_call_fn_764703iuvw?б<
5б2
$і!
inputs         
 

 
p 

 
ф "!і
unknown          я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765442ћlmnђб}
vбs
 і
inputs         
KбH
"і
states_0          
"і
states_1          
p
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765474ћlmnђб}
vбs
 і
inputs         
KбH
"і
states_0          
"і
states_1          
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765540ћopqђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765572ћopqђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765638ћrstђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765670ћrstђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765736ћuvwђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_765768ћuvwђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0          
SџP
&і#
tensor_0_1_0          
&і#
tensor_0_1_1          
џ ▒
*__inference_lstm_cell_layer_call_fn_765393ѓlmnђб}
vбs
 і
inputs         
KбH
"і
states_0          
"і
states_1          
p
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765410ѓlmnђб}
vбs
 і
inputs         
KбH
"і
states_0          
"і
states_1          
p 
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765491ѓopqђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765508ѓopqђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765589ѓrstђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765606ѓrstђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765687ѓuvwђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          ▒
*__inference_lstm_cell_layer_call_fn_765704ѓuvwђб}
vбs
 і
inputs          
KбH
"і
states_0          
"і
states_1          
p 
ф "xбu
"і
tensor_0          
OџL
$і!

tensor_1_0          
$і!

tensor_1_1          м
I__inference_sequential_11_layer_call_and_return_conditional_losses_761814ёlmnopqrstuvw[\jkBб?
8б5
+і(
lstm_35_input         

p

 
ф ",б)
"і
tensor_0         
џ м
I__inference_sequential_11_layer_call_and_return_conditional_losses_762460ёlmnopqrstuvw[\jkBб?
8б5
+і(
lstm_35_input         

p 

 
ф ",б)
"і
tensor_0         
џ Ф
.__inference_sequential_11_layer_call_fn_762497ylmnopqrstuvw[\jkBб?
8б5
+і(
lstm_35_input         

p

 
ф "!і
unknown         Ф
.__inference_sequential_11_layer_call_fn_762534ylmnopqrstuvw[\jkBб?
8б5
+і(
lstm_35_input         

p 

 
ф "!і
unknown         й
$__inference_signature_wrapper_762730ћlmnopqrstuvw[\jkKбH
б 
Aф>
<
lstm_35_input+і(
lstm_35_input         
"3ф0
.
dense_32"і
dense_32         