╩▄1
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
ѕ"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8 ─-
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
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_22/bias
y
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_22/bias
y
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes
:*
dtype0
ѕ
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_22/kernel
Ђ
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel*
_output_shapes

:*
dtype0
ѕ
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_22/kernel
Ђ
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel*
_output_shapes

:*
dtype0
ђ
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_21/bias
y
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_21/bias
y
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes
:*
dtype0
ѕ
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_21/kernel
Ђ
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes

:*
dtype0
ѕ
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_21/kernel
Ђ
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes

:*
dtype0
ђ
Adam/v/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_20/bias
y
(Adam/v/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_20/bias
y
(Adam/m/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/bias*
_output_shapes
:*
dtype0
ѕ
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_20/kernel
Ђ
*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes

:*
dtype0
ѕ
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_20/kernel
Ђ
*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
_output_shapes

:*
dtype0
ђ
Adam/v/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_19/bias
y
(Adam/v/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_19/bias
y
(Adam/m/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/bias*
_output_shapes
:*
dtype0
ѕ
Adam/v/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_19/kernel
Ђ
*Adam/v/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/kernel*
_output_shapes

:*
dtype0
ѕ
Adam/m/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_19/kernel
Ђ
*Adam/m/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/kernel*
_output_shapes

:*
dtype0
њ
Adam/v/lstm_23/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/v/lstm_23/lstm_cell/bias
І
1Adam/v/lstm_23/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_23/lstm_cell/bias*
_output_shapes
:@*
dtype0
њ
Adam/m/lstm_23/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/m/lstm_23/lstm_cell/bias
І
1Adam/m/lstm_23/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_23/lstm_cell/bias*
_output_shapes
:@*
dtype0
«
)Adam/v/lstm_23/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/v/lstm_23/lstm_cell/recurrent_kernel
Д
=Adam/v/lstm_23/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_23/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
«
)Adam/m/lstm_23/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/m/lstm_23/lstm_cell/recurrent_kernel
Д
=Adam/m/lstm_23/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_23/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
џ
Adam/v/lstm_23/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/v/lstm_23/lstm_cell/kernel
Њ
3Adam/v/lstm_23/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_23/lstm_cell/kernel*
_output_shapes

:@*
dtype0
џ
Adam/m/lstm_23/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/m/lstm_23/lstm_cell/kernel
Њ
3Adam/m/lstm_23/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_23/lstm_cell/kernel*
_output_shapes

:@*
dtype0
њ
Adam/v/lstm_22/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/v/lstm_22/lstm_cell/bias
І
1Adam/v/lstm_22/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_22/lstm_cell/bias*
_output_shapes
:@*
dtype0
њ
Adam/m/lstm_22/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/m/lstm_22/lstm_cell/bias
І
1Adam/m/lstm_22/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_22/lstm_cell/bias*
_output_shapes
:@*
dtype0
«
)Adam/v/lstm_22/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/v/lstm_22/lstm_cell/recurrent_kernel
Д
=Adam/v/lstm_22/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_22/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
«
)Adam/m/lstm_22/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/m/lstm_22/lstm_cell/recurrent_kernel
Д
=Adam/m/lstm_22/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_22/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
џ
Adam/v/lstm_22/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/v/lstm_22/lstm_cell/kernel
Њ
3Adam/v/lstm_22/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_22/lstm_cell/kernel*
_output_shapes

:@*
dtype0
џ
Adam/m/lstm_22/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/m/lstm_22/lstm_cell/kernel
Њ
3Adam/m/lstm_22/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_22/lstm_cell/kernel*
_output_shapes

:@*
dtype0
њ
Adam/v/lstm_21/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/v/lstm_21/lstm_cell/bias
І
1Adam/v/lstm_21/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_21/lstm_cell/bias*
_output_shapes
:@*
dtype0
њ
Adam/m/lstm_21/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/m/lstm_21/lstm_cell/bias
І
1Adam/m/lstm_21/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_21/lstm_cell/bias*
_output_shapes
:@*
dtype0
«
)Adam/v/lstm_21/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/v/lstm_21/lstm_cell/recurrent_kernel
Д
=Adam/v/lstm_21/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_21/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
«
)Adam/m/lstm_21/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/m/lstm_21/lstm_cell/recurrent_kernel
Д
=Adam/m/lstm_21/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_21/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
џ
Adam/v/lstm_21/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/v/lstm_21/lstm_cell/kernel
Њ
3Adam/v/lstm_21/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_21/lstm_cell/kernel*
_output_shapes

:@*
dtype0
џ
Adam/m/lstm_21/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/m/lstm_21/lstm_cell/kernel
Њ
3Adam/m/lstm_21/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_21/lstm_cell/kernel*
_output_shapes

:@*
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
ё
lstm_23/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namelstm_23/lstm_cell/bias
}
*lstm_23/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell/bias*
_output_shapes
:@*
dtype0
а
"lstm_23/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"lstm_23/lstm_cell/recurrent_kernel
Ў
6lstm_23/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_23/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
ї
lstm_23/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_namelstm_23/lstm_cell/kernel
Ё
,lstm_23/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell/kernel*
_output_shapes

:@*
dtype0
ё
lstm_22/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namelstm_22/lstm_cell/bias
}
*lstm_22/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell/bias*
_output_shapes
:@*
dtype0
а
"lstm_22/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"lstm_22/lstm_cell/recurrent_kernel
Ў
6lstm_22/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_22/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
ї
lstm_22/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_namelstm_22/lstm_cell/kernel
Ё
,lstm_22/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell/kernel*
_output_shapes

:@*
dtype0
ё
lstm_21/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namelstm_21/lstm_cell/bias
}
*lstm_21/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell/bias*
_output_shapes
:@*
dtype0
а
"lstm_21/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"lstm_21/lstm_cell/recurrent_kernel
Ў
6lstm_21/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_21/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
ї
lstm_21/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_namelstm_21/lstm_cell/kernel
Ё
,lstm_21/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell/kernel*
_output_shapes

:@*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:*
dtype0
ѕ
serving_default_lstm_21_inputPlaceholder*+
_output_shapes
:         
*
dtype0* 
shape:         

▄
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_21_inputlstm_21/lstm_cell/kernel"lstm_21/lstm_cell/recurrent_kernellstm_21/lstm_cell/biaslstm_22/lstm_cell/kernel"lstm_22/lstm_cell/recurrent_kernellstm_22/lstm_cell/biaslstm_23/lstm_cell/kernel"lstm_23/lstm_cell/recurrent_kernellstm_23/lstm_cell/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_508518

NoOpNoOp
тЇ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЪЇ
valueћЇBљЇ BѕЇ
«
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
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
Ц
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator* 
┴
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator
.cell
/
state_spec*
Ц
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
┴
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator
>cell
?
state_spec*
Ц
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator* 
д
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias*
Ц
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator* 
д
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
Ц
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d_random_generator* 
д
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias*
Ц
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator* 
д
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias*
Є
|0
}1
~2
3
ђ4
Ђ5
ѓ6
Ѓ7
ё8
M9
N10
\11
]12
k13
l14
z15
{16*
Є
|0
}1
~2
3
ђ4
Ђ5
ѓ6
Ѓ7
ё8
M9
N10
\11
]12
k13
l14
z15
{16*
* 
х
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

іtrace_0
Іtrace_1* 

їtrace_0
Їtrace_1* 
* 
ѕ
ј
_variables
Ј_iterations
љ_learning_rate
Љ_index_dict
њ
_momentums
Њ_velocities
ћ_update_step_xla*

Ћserving_default* 

|0
}1
~2*

|0
}1
~2*
* 
Ц
ќstates
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
юtrace_0
Юtrace_1
ъtrace_2
Ъtrace_3* 
:
аtrace_0
Аtrace_1
бtrace_2
Бtrace_3* 
* 
в
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
ф_random_generator
Ф
state_size

|kernel
}recurrent_kernel
~bias*
* 
* 
* 
* 
ќ
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

▒trace_0
▓trace_1* 

│trace_0
┤trace_1* 
* 

0
ђ1
Ђ2*

0
ђ1
Ђ2*
* 
Ц
хstates
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
:
╗trace_0
╝trace_1
йtrace_2
Йtrace_3* 
:
┐trace_0
└trace_1
┴trace_2
┬trace_3* 
* 
ь
├	variables
─trainable_variables
┼regularization_losses
к	keras_api
К__call__
+╚&call_and_return_all_conditional_losses
╔_random_generator
╩
state_size

kernel
ђrecurrent_kernel
	Ђbias*
* 
* 
* 
* 
ќ
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

лtrace_0
Лtrace_1* 

мtrace_0
Мtrace_1* 
* 

ѓ0
Ѓ1
ё2*

ѓ0
Ѓ1
ё2*
* 
Ц
нstates
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
:
┌trace_0
█trace_1
▄trace_2
Пtrace_3* 
:
яtrace_0
▀trace_1
Яtrace_2
рtrace_3* 
* 
Ь
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses
У_random_generator
ж
state_size
ѓkernel
Ѓrecurrent_kernel
	ёbias*
* 
* 
* 
* 
ќ
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

№trace_0
­trace_1* 

ыtrace_0
Ыtrace_1* 
* 

M0
N1*

M0
N1*
* 
ў
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

Эtrace_0* 

щtrace_0* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
Щnon_trainable_variables
чlayers
Чmetrics
 §layer_regularization_losses
■layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

 trace_0
ђtrace_1* 

Ђtrace_0
ѓtrace_1* 
* 

\0
]1*

\0
]1*
* 
ў
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

ѕtrace_0* 

Ѕtrace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

Јtrace_0
љtrace_1* 

Љtrace_0
њtrace_1* 
* 

k0
l1*

k0
l1*
* 
ў
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

ўtrace_0* 

Ўtrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

Ъtrace_0
аtrace_1* 

Аtrace_0
бtrace_1* 
* 

z0
{1*

z0
{1*
* 
ў
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

еtrace_0* 

Еtrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_21/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_21/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_21/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_22/lstm_cell/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_22/lstm_cell/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_22/lstm_cell/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_23/lstm_cell/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_23/lstm_cell/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_23/lstm_cell/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
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
10
11
12*

ф0
Ф1*
* 
* 
* 
* 
* 
* 
х
Ј0
г1
Г2
«3
»4
░5
▒6
▓7
│8
┤9
х10
Х11
и12
И13
╣14
║15
╗16
╝17
й18
Й19
┐20
└21
┴22
┬23
├24
─25
┼26
к27
К28
╚29
╔30
╩31
╦32
╠33
═34*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Њ
г0
«1
░2
▓3
┤4
Х5
И6
║7
╝8
Й9
└10
┬11
─12
к13
╚14
╩15
╠16*
Њ
Г0
»1
▒2
│3
х4
и5
╣6
╗7
й8
┐9
┴10
├11
┼12
К13
╔14
╦15
═16*
* 
* 
* 
* 

0*
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
|0
}1
~2*

|0
}1
~2*
* 
ъ
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*

Мtrace_0
нtrace_1* 

Нtrace_0
оtrace_1* 
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

.0*
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

0
ђ1
Ђ2*

0
ђ1
Ђ2*
* 
ъ
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
├	variables
─trainable_variables
┼regularization_losses
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses*

▄trace_0
Пtrace_1* 

яtrace_0
▀trace_1* 
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

>0*
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

ѓ0
Ѓ1
ё2*

ѓ0
Ѓ1
ё2*
* 
ъ
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses*

тtrace_0
Тtrace_1* 

уtrace_0
Уtrace_1* 
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
ж	variables
Ж	keras_api

вtotal

Вcount*
M
ь	variables
Ь	keras_api

№total

­count
ы
_fn_kwargs*
jd
VARIABLE_VALUEAdam/m/lstm_21/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_21/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/lstm_21/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/lstm_21/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/lstm_21/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/lstm_21/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/lstm_22/lstm_cell/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_22/lstm_cell/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/lstm_22/lstm_cell/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/lstm_22/lstm_cell/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_22/lstm_cell/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_22/lstm_cell/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/lstm_23/lstm_cell/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/lstm_23/lstm_cell/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/lstm_23/lstm_cell/recurrent_kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/lstm_23/lstm_cell/recurrent_kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_23/lstm_cell/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_23/lstm_cell/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_19/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_19/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_19/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_19/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_20/kernel2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_20/kernel2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_20/bias2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_20/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_21/kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_21/kernel2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_21/bias2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_21/bias2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_22/kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
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
в0
В1*

ж	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

№0
­1*

ь	variables*
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
ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biaslstm_21/lstm_cell/kernel"lstm_21/lstm_cell/recurrent_kernellstm_21/lstm_cell/biaslstm_22/lstm_cell/kernel"lstm_22/lstm_cell/recurrent_kernellstm_22/lstm_cell/biaslstm_23/lstm_cell/kernel"lstm_23/lstm_cell/recurrent_kernellstm_23/lstm_cell/bias	iterationlearning_rateAdam/m/lstm_21/lstm_cell/kernelAdam/v/lstm_21/lstm_cell/kernel)Adam/m/lstm_21/lstm_cell/recurrent_kernel)Adam/v/lstm_21/lstm_cell/recurrent_kernelAdam/m/lstm_21/lstm_cell/biasAdam/v/lstm_21/lstm_cell/biasAdam/m/lstm_22/lstm_cell/kernelAdam/v/lstm_22/lstm_cell/kernel)Adam/m/lstm_22/lstm_cell/recurrent_kernel)Adam/v/lstm_22/lstm_cell/recurrent_kernelAdam/m/lstm_22/lstm_cell/biasAdam/v/lstm_22/lstm_cell/biasAdam/m/lstm_23/lstm_cell/kernelAdam/v/lstm_23/lstm_cell/kernel)Adam/m/lstm_23/lstm_cell/recurrent_kernel)Adam/v/lstm_23/lstm_cell/recurrent_kernelAdam/m/lstm_23/lstm_cell/biasAdam/v/lstm_23/lstm_cell/biasAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biastotal_1count_1totalcountConst*F
Tin?
=2;*
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
__inference__traced_save_511273
§
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biaslstm_21/lstm_cell/kernel"lstm_21/lstm_cell/recurrent_kernellstm_21/lstm_cell/biaslstm_22/lstm_cell/kernel"lstm_22/lstm_cell/recurrent_kernellstm_22/lstm_cell/biaslstm_23/lstm_cell/kernel"lstm_23/lstm_cell/recurrent_kernellstm_23/lstm_cell/bias	iterationlearning_rateAdam/m/lstm_21/lstm_cell/kernelAdam/v/lstm_21/lstm_cell/kernel)Adam/m/lstm_21/lstm_cell/recurrent_kernel)Adam/v/lstm_21/lstm_cell/recurrent_kernelAdam/m/lstm_21/lstm_cell/biasAdam/v/lstm_21/lstm_cell/biasAdam/m/lstm_22/lstm_cell/kernelAdam/v/lstm_22/lstm_cell/kernel)Adam/m/lstm_22/lstm_cell/recurrent_kernel)Adam/v/lstm_22/lstm_cell/recurrent_kernelAdam/m/lstm_22/lstm_cell/biasAdam/v/lstm_22/lstm_cell/biasAdam/m/lstm_23/lstm_cell/kernelAdam/v/lstm_23/lstm_cell/kernel)Adam/m/lstm_23/lstm_cell/recurrent_kernel)Adam/v/lstm_23/lstm_cell/recurrent_kernelAdam/m/lstm_23/lstm_cell/biasAdam/v/lstm_23/lstm_cell/biasAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biastotal_1count_1totalcount*E
Tin>
<2:*
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
"__inference__traced_restore_511453йџ+
┬	
├
while_cond_510052
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_510052___redundant_placeholder04
0while_while_cond_510052___redundant_placeholder14
0while_while_cond_510052___redundant_placeholder24
0while_while_cond_510052___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
кJ
є
C__inference_lstm_23_layer_call_and_return_conditional_losses_508183

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508098*
condR
while_cond_508097*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
п%
╚
while_body_507001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_507025_0:@*
while_lstm_cell_507027_0:@&
while_lstm_cell_507029_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_507025:@(
while_lstm_cell_507027:@$
while_lstm_cell_507029:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_507025_0while_lstm_cell_507027_0while_lstm_cell_507029_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506986r
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_507025while_lstm_cell_507025_0"2
while_lstm_cell_507027while_lstm_cell_507027_0"2
while_lstm_cell_507029while_lstm_cell_507029_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name507029:&	"
 
_user_specified_name507027:&"
 
_user_specified_name507025:_[
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
:         :-)
'
_output_shapes
:         :
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
)__inference_dense_19_layer_call_fn_510464

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_507631o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510460:&"
 
_user_specified_name510458:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦

ш
D__inference_dense_19_layer_call_and_return_conditional_losses_507631

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
└I
є
C__inference_lstm_21_layer_call_and_return_conditional_losses_507869

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507785*
condR
while_cond_507784*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
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
Х

e
F__inference_dropout_34_layer_call_and_return_conditional_losses_509799

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
Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
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
T
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
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Х

e
F__inference_dropout_34_layer_call_and_return_conditional_losses_507454

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
Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
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
T
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
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
П8
»
while_body_509264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510713

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
П8
»
while_body_509050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
щ9
»
while_body_508098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
┐E
В
H__inference_sequential_7_layer_call_and_return_conditional_losses_507724
lstm_21_input 
lstm_21_507273:@ 
lstm_21_507275:@
lstm_21_507277:@ 
lstm_22_507436:@ 
lstm_22_507438:@
lstm_22_507440:@ 
lstm_23_507601:@ 
lstm_23_507603:@
lstm_23_507605:@!
dense_19_507632:
dense_19_507634:!
dense_20_507661:
dense_20_507663:!
dense_21_507690:
dense_21_507692:!
dense_22_507718:
dense_22_507720:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб"dropout_33/StatefulPartitionedCallб"dropout_34/StatefulPartitionedCallб"dropout_35/StatefulPartitionedCallб"dropout_36/StatefulPartitionedCallб"dropout_37/StatefulPartitionedCallб"dropout_38/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЅ
lstm_21/StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputlstm_21_507273lstm_21_507275lstm_21_507277*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_507272Ы
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_507291Д
lstm_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_33/StatefulPartitionedCall:output:0lstm_22_507436lstm_22_507438lstm_22_507440*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_507435Ќ
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_507454Б
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0lstm_23_507601lstm_23_507603lstm_23_507605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_507600Њ
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_507619Ћ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0dense_19_507632dense_19_507634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_507631ћ
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_35/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_507648Ћ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0dense_20_507661dense_20_507663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_507660ћ
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_507677Ћ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_21_507690dense_21_507692*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_507689ћ
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_507706Ћ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_22_507718dense_22_507720*
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
D__inference_dense_22_layer_call_and_return_conditional_losses_507717x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:&"
 
_user_specified_name507720:&"
 
_user_specified_name507718:&"
 
_user_specified_name507692:&"
 
_user_specified_name507690:&"
 
_user_specified_name507663:&"
 
_user_specified_name507661:&"
 
_user_specified_name507634:&
"
 
_user_specified_name507632:&	"
 
_user_specified_name507605:&"
 
_user_specified_name507603:&"
 
_user_specified_name507601:&"
 
_user_specified_name507440:&"
 
_user_specified_name507438:&"
 
_user_specified_name507436:&"
 
_user_specified_name507277:&"
 
_user_specified_name507275:&"
 
_user_specified_name507273:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_21_input
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510877

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Я
┤
(__inference_lstm_23_layer_call_fn_509826
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_507071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509822:&"
 
_user_specified_name509820:&"
 
_user_specified_name509818:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
┬	
├
while_cond_507940
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507940___redundant_placeholder04
0while_while_cond_507940___redundant_placeholder14
0while_while_cond_507940___redundant_placeholder24
0while_while_cond_507940___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
П8
»
while_body_509407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
╬
у
&sequential_7_lstm_21_while_cond_505687F
Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counterL
Hsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations*
&sequential_7_lstm_21_while_placeholder,
(sequential_7_lstm_21_while_placeholder_1,
(sequential_7_lstm_21_while_placeholder_2,
(sequential_7_lstm_21_while_placeholder_3H
Dsequential_7_lstm_21_while_less_sequential_7_lstm_21_strided_slice_1^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_505687___redundant_placeholder0^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_505687___redundant_placeholder1^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_505687___redundant_placeholder2^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_505687___redundant_placeholder3'
#sequential_7_lstm_21_while_identity
Х
sequential_7/lstm_21/while/LessLess&sequential_7_lstm_21_while_placeholderDsequential_7_lstm_21_while_less_sequential_7_lstm_21_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_21/while/IdentityIdentity#sequential_7/lstm_21/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_21_while_identity,sequential_7/lstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::

_output_shapes
::\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_21/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_21/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_21/while/loop_counter
џ

e
F__inference_dropout_35_layer_call_and_return_conditional_losses_507619

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510909

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
П8
»
while_body_509550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_508217

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л
▓
(__inference_lstm_22_layer_call_fn_509205

inputs
unknown:@
	unknown_0:@
	unknown_1:@
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
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_508025s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509201:&"
 
_user_specified_name509199:&"
 
_user_specified_name509197:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
жJ
ѕ
C__inference_lstm_23_layer_call_and_return_conditional_losses_510138
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_510053*
condR
while_cond_510052*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
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
 :                  
"
_user_specified_name
inputs_0
¤
d
+__inference_dropout_35_layer_call_fn_510433

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_507619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
■I
ѕ
C__inference_lstm_21_layer_call_and_return_conditional_losses_508848
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508764*
condR
while_cond_508763*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
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
┬	
├
while_cond_508620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_508620___redundant_placeholder04
0while_while_cond_508620___redundant_placeholder14
0while_while_cond_508620___redundant_placeholder24
0while_while_cond_508620___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
э	
ш
D__inference_dense_22_layer_call_and_return_conditional_losses_507717

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:         : : 20
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
:         
 
_user_specified_nameinputs
Х
­
*__inference_lstm_cell_layer_call_fn_510828

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510820:&"
 
_user_specified_name510818:&"
 
_user_specified_name510816:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_506305
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_506305___redundant_placeholder04
0while_while_cond_506305___redundant_placeholder14
0while_while_cond_506305___redundant_placeholder24
0while_while_cond_506305___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_33_layer_call_and_return_conditional_losses_507291

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
Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
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
T
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
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Щ
┤
(__inference_lstm_22_layer_call_fn_509172
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_506576|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509168:&"
 
_user_specified_name509166:&"
 
_user_specified_name509164:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╝$
╚
while_body_506306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_506330_0:@*
while_lstm_cell_506332_0:@&
while_lstm_cell_506334_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_506330:@(
while_lstm_cell_506332:@$
while_lstm_cell_506334:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_506330_0while_lstm_cell_506332_0while_lstm_cell_506334_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506292┘
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_506330while_lstm_cell_506330_0"2
while_lstm_cell_506332while_lstm_cell_506332_0"2
while_lstm_cell_506334while_lstm_cell_506334_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name506334:&	"
 
_user_specified_name506332:&"
 
_user_specified_name506330:_[
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
:         :-)
'
_output_shapes
:         :
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
щ9
»
while_body_510198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_510549

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х8
Ш
C__inference_lstm_22_layer_call_and_return_conditional_losses_506721

inputs"
lstm_cell_506639:@"
lstm_cell_506641:@
lstm_cell_506643:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506639lstm_cell_506641lstm_cell_506643*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506638n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506639lstm_cell_506641lstm_cell_506643*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506652*
condR
while_cond_506651*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506643:&"
 
_user_specified_name506641:&"
 
_user_specified_name506639:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
А
G
+__inference_dropout_38_layer_call_fn_510579

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_508228`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╚
▓
(__inference_lstm_23_layer_call_fn_509848

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_508183o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509844:&"
 
_user_specified_name509842:&"
 
_user_specified_name509840:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
└I
є
C__inference_lstm_21_layer_call_and_return_conditional_losses_508991

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508907*
condR
while_cond_508906*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
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
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510779

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_510569

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
▒
G
+__inference_dropout_34_layer_call_fn_509787

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
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_508037d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▒
G
+__inference_dropout_33_layer_call_fn_509144

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
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_507881d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
џ

e
F__inference_dropout_37_layer_call_and_return_conditional_losses_507677

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_510591

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_508097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_508097___redundant_placeholder04
0while_while_cond_508097___redundant_placeholder14
0while_while_cond_508097___redundant_placeholder24
0while_while_cond_508097___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
╦

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_507689

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
┬	
├
while_cond_506651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_506651___redundant_placeholder04
0while_while_cond_506651___redundant_placeholder14
0while_while_cond_506651___redundant_placeholder24
0while_while_cond_506651___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
П8
»
while_body_507785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
з
┬
-__inference_sequential_7_layer_call_fn_508275
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_507724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508271:&"
 
_user_specified_name508269:&"
 
_user_specified_name508267:&"
 
_user_specified_name508265:&"
 
_user_specified_name508263:&"
 
_user_specified_name508261:&"
 
_user_specified_name508259:&
"
 
_user_specified_name508257:&	"
 
_user_specified_name508255:&"
 
_user_specified_name508253:&"
 
_user_specified_name508251:&"
 
_user_specified_name508249:&"
 
_user_specified_name508247:&"
 
_user_specified_name508245:&"
 
_user_specified_name508243:&"
 
_user_specified_name508241:&"
 
_user_specified_name508239:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_21_input
љR
л
&sequential_7_lstm_22_while_body_505828F
Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counterL
Hsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations*
&sequential_7_lstm_22_while_placeholder,
(sequential_7_lstm_22_while_placeholder_1,
(sequential_7_lstm_22_while_placeholder_2,
(sequential_7_lstm_22_while_placeholder_3E
Asequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1_0Ђ
}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0W
Esequential_7_lstm_22_while_lstm_cell_matmul_readvariableop_resource_0:@Y
Gsequential_7_lstm_22_while_lstm_cell_matmul_1_readvariableop_resource_0:@T
Fsequential_7_lstm_22_while_lstm_cell_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_22_while_identity)
%sequential_7_lstm_22_while_identity_1)
%sequential_7_lstm_22_while_identity_2)
%sequential_7_lstm_22_while_identity_3)
%sequential_7_lstm_22_while_identity_4)
%sequential_7_lstm_22_while_identity_5C
?sequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1
{sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensorU
Csequential_7_lstm_22_while_lstm_cell_matmul_readvariableop_resource:@W
Esequential_7_lstm_22_while_lstm_cell_matmul_1_readvariableop_resource:@R
Dsequential_7_lstm_22_while_lstm_cell_biasadd_readvariableop_resource:@ѕб;sequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOpб:sequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOpб<sequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_22_while_placeholderUsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0└
:sequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpEsequential_7_lstm_22_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ы
+sequential_7/lstm_22/while/lstm_cell/MatMulMatMulEsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @─
<sequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpGsequential_7_lstm_22_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0┘
-sequential_7/lstm_22/while/lstm_cell/MatMul_1MatMul(sequential_7_lstm_22_while_placeholder_2Dsequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @М
(sequential_7/lstm_22/while/lstm_cell/addAddV25sequential_7/lstm_22/while/lstm_cell/MatMul:product:07sequential_7/lstm_22/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @Й
;sequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpFsequential_7_lstm_22_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0▄
,sequential_7/lstm_22/while/lstm_cell/BiasAddBiasAdd,sequential_7/lstm_22/while/lstm_cell/add:z:0Csequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
4sequential_7/lstm_22/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_7/lstm_22/while/lstm_cell/splitSplit=sequential_7/lstm_22/while/lstm_cell/split/split_dim:output:05sequential_7/lstm_22/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitъ
,sequential_7/lstm_22/while/lstm_cell/SigmoidSigmoid3sequential_7/lstm_22/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_22/while/lstm_cell/Sigmoid_1Sigmoid3sequential_7/lstm_22/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ┐
(sequential_7/lstm_22/while/lstm_cell/mulMul2sequential_7/lstm_22/while/lstm_cell/Sigmoid_1:y:0(sequential_7_lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         ў
)sequential_7/lstm_22/while/lstm_cell/ReluRelu3sequential_7/lstm_22/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╬
*sequential_7/lstm_22/while/lstm_cell/mul_1Mul0sequential_7/lstm_22/while/lstm_cell/Sigmoid:y:07sequential_7/lstm_22/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ├
*sequential_7/lstm_22/while/lstm_cell/add_1AddV2,sequential_7/lstm_22/while/lstm_cell/mul:z:0.sequential_7/lstm_22/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_22/while/lstm_cell/Sigmoid_2Sigmoid3sequential_7/lstm_22/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ћ
+sequential_7/lstm_22/while/lstm_cell/Relu_1Relu.sequential_7/lstm_22/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         м
*sequential_7/lstm_22/while/lstm_cell/mul_2Mul2sequential_7/lstm_22/while/lstm_cell/Sigmoid_2:y:09sequential_7/lstm_22/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ќ
?sequential_7/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_22_while_placeholder_1&sequential_7_lstm_22_while_placeholder.sequential_7/lstm_22/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_22/while/addAddV2&sequential_7_lstm_22_while_placeholder)sequential_7/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_22/while/add_1AddV2Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counter+sequential_7/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_22/while/IdentityIdentity$sequential_7/lstm_22/while/add_1:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_22/while/Identity_1IdentityHsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_22/while/Identity_2Identity"sequential_7/lstm_22/while/add:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_22/while/Identity_3IdentityOsequential_7/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: х
%sequential_7/lstm_22/while/Identity_4Identity.sequential_7/lstm_22/while/lstm_cell/mul_2:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*'
_output_shapes
:         х
%sequential_7/lstm_22/while/Identity_5Identity.sequential_7/lstm_22/while/lstm_cell/add_1:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*'
_output_shapes
:         э
sequential_7/lstm_22/while/NoOpNoOp<^sequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOp;^sequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOp=^sequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "W
%sequential_7_lstm_22_while_identity_1.sequential_7/lstm_22/while/Identity_1:output:0"W
%sequential_7_lstm_22_while_identity_2.sequential_7/lstm_22/while/Identity_2:output:0"W
%sequential_7_lstm_22_while_identity_3.sequential_7/lstm_22/while/Identity_3:output:0"W
%sequential_7_lstm_22_while_identity_4.sequential_7/lstm_22/while/Identity_4:output:0"W
%sequential_7_lstm_22_while_identity_5.sequential_7/lstm_22/while/Identity_5:output:0"S
#sequential_7_lstm_22_while_identity,sequential_7/lstm_22/while/Identity:output:0"ј
Dsequential_7_lstm_22_while_lstm_cell_biasadd_readvariableop_resourceFsequential_7_lstm_22_while_lstm_cell_biasadd_readvariableop_resource_0"љ
Esequential_7_lstm_22_while_lstm_cell_matmul_1_readvariableop_resourceGsequential_7_lstm_22_while_lstm_cell_matmul_1_readvariableop_resource_0"ї
Csequential_7_lstm_22_while_lstm_cell_matmul_readvariableop_resourceEsequential_7_lstm_22_while_lstm_cell_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1Asequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1_0"Ч
{sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2z
;sequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOp;sequential_7/lstm_22/while/lstm_cell/BiasAdd/ReadVariableOp2x
:sequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOp:sequential_7/lstm_22/while/lstm_cell/MatMul/ReadVariableOp2|
<sequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOp<sequential_7/lstm_22/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:tp

_output_shapes
: 
V
_user_specified_name><sequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor:\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_22/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_22/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_22/while/loop_counter
јІ
и%
"__inference__traced_restore_511453
file_prefix2
 assignvariableop_dense_19_kernel:.
 assignvariableop_1_dense_19_bias:4
"assignvariableop_2_dense_20_kernel:.
 assignvariableop_3_dense_20_bias:4
"assignvariableop_4_dense_21_kernel:.
 assignvariableop_5_dense_21_bias:4
"assignvariableop_6_dense_22_kernel:.
 assignvariableop_7_dense_22_bias:=
+assignvariableop_8_lstm_21_lstm_cell_kernel:@G
5assignvariableop_9_lstm_21_lstm_cell_recurrent_kernel:@8
*assignvariableop_10_lstm_21_lstm_cell_bias:@>
,assignvariableop_11_lstm_22_lstm_cell_kernel:@H
6assignvariableop_12_lstm_22_lstm_cell_recurrent_kernel:@8
*assignvariableop_13_lstm_22_lstm_cell_bias:@>
,assignvariableop_14_lstm_23_lstm_cell_kernel:@H
6assignvariableop_15_lstm_23_lstm_cell_recurrent_kernel:@8
*assignvariableop_16_lstm_23_lstm_cell_bias:@'
assignvariableop_17_iteration:	 +
!assignvariableop_18_learning_rate: E
3assignvariableop_19_adam_m_lstm_21_lstm_cell_kernel:@E
3assignvariableop_20_adam_v_lstm_21_lstm_cell_kernel:@O
=assignvariableop_21_adam_m_lstm_21_lstm_cell_recurrent_kernel:@O
=assignvariableop_22_adam_v_lstm_21_lstm_cell_recurrent_kernel:@?
1assignvariableop_23_adam_m_lstm_21_lstm_cell_bias:@?
1assignvariableop_24_adam_v_lstm_21_lstm_cell_bias:@E
3assignvariableop_25_adam_m_lstm_22_lstm_cell_kernel:@E
3assignvariableop_26_adam_v_lstm_22_lstm_cell_kernel:@O
=assignvariableop_27_adam_m_lstm_22_lstm_cell_recurrent_kernel:@O
=assignvariableop_28_adam_v_lstm_22_lstm_cell_recurrent_kernel:@?
1assignvariableop_29_adam_m_lstm_22_lstm_cell_bias:@?
1assignvariableop_30_adam_v_lstm_22_lstm_cell_bias:@E
3assignvariableop_31_adam_m_lstm_23_lstm_cell_kernel:@E
3assignvariableop_32_adam_v_lstm_23_lstm_cell_kernel:@O
=assignvariableop_33_adam_m_lstm_23_lstm_cell_recurrent_kernel:@O
=assignvariableop_34_adam_v_lstm_23_lstm_cell_recurrent_kernel:@?
1assignvariableop_35_adam_m_lstm_23_lstm_cell_bias:@?
1assignvariableop_36_adam_v_lstm_23_lstm_cell_bias:@<
*assignvariableop_37_adam_m_dense_19_kernel:<
*assignvariableop_38_adam_v_dense_19_kernel:6
(assignvariableop_39_adam_m_dense_19_bias:6
(assignvariableop_40_adam_v_dense_19_bias:<
*assignvariableop_41_adam_m_dense_20_kernel:<
*assignvariableop_42_adam_v_dense_20_kernel:6
(assignvariableop_43_adam_m_dense_20_bias:6
(assignvariableop_44_adam_v_dense_20_bias:<
*assignvariableop_45_adam_m_dense_21_kernel:<
*assignvariableop_46_adam_v_dense_21_kernel:6
(assignvariableop_47_adam_m_dense_21_bias:6
(assignvariableop_48_adam_v_dense_21_bias:<
*assignvariableop_49_adam_m_dense_22_kernel:<
*assignvariableop_50_adam_v_dense_22_kernel:6
(assignvariableop_51_adam_m_dense_22_bias:6
(assignvariableop_52_adam_v_dense_22_bias:%
assignvariableop_53_total_1: %
assignvariableop_54_count_1: #
assignvariableop_55_total: #
assignvariableop_56_count: 
identity_58ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9╦
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*ы
valueуBС:B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHт
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Є
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ├
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*■
_output_shapesв
У::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_19_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_19_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_20_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_20_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_21_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_21_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_22_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_22_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_8AssignVariableOp+assignvariableop_8_lstm_21_lstm_cell_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_9AssignVariableOp5assignvariableop_9_lstm_21_lstm_cell_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_10AssignVariableOp*assignvariableop_10_lstm_21_lstm_cell_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_11AssignVariableOp,assignvariableop_11_lstm_22_lstm_cell_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_12AssignVariableOp6assignvariableop_12_lstm_22_lstm_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_13AssignVariableOp*assignvariableop_13_lstm_22_lstm_cell_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_14AssignVariableOp,assignvariableop_14_lstm_23_lstm_cell_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_15AssignVariableOp6assignvariableop_15_lstm_23_lstm_cell_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_16AssignVariableOp*assignvariableop_16_lstm_23_lstm_cell_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_17AssignVariableOpassignvariableop_17_iterationIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_18AssignVariableOp!assignvariableop_18_learning_rateIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_m_lstm_21_lstm_cell_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_v_lstm_21_lstm_cell_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_21AssignVariableOp=assignvariableop_21_adam_m_lstm_21_lstm_cell_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_22AssignVariableOp=assignvariableop_22_adam_v_lstm_21_lstm_cell_recurrent_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_m_lstm_21_lstm_cell_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_v_lstm_21_lstm_cell_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_m_lstm_22_lstm_cell_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_v_lstm_22_lstm_cell_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_m_lstm_22_lstm_cell_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_28AssignVariableOp=assignvariableop_28_adam_v_lstm_22_lstm_cell_recurrent_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_m_lstm_22_lstm_cell_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_v_lstm_22_lstm_cell_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_m_lstm_23_lstm_cell_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_v_lstm_23_lstm_cell_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_33AssignVariableOp=assignvariableop_33_adam_m_lstm_23_lstm_cell_recurrent_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_34AssignVariableOp=assignvariableop_34_adam_v_lstm_23_lstm_cell_recurrent_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_m_lstm_23_lstm_cell_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_v_lstm_23_lstm_cell_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_m_dense_19_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_v_dense_19_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_m_dense_19_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_v_dense_19_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_m_dense_20_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_v_dense_20_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_m_dense_20_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_v_dense_20_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_m_dense_21_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_v_dense_21_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_m_dense_21_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_v_dense_21_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_m_dense_22_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_v_dense_22_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_m_dense_22_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_v_dense_22_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 х

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ■	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_58Identity_58:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%9!

_user_specified_namecount:%8!

_user_specified_nametotal:'7#
!
_user_specified_name	count_1:'6#
!
_user_specified_name	total_1:450
.
_user_specified_nameAdam/v/dense_22/bias:440
.
_user_specified_nameAdam/m/dense_22/bias:632
0
_user_specified_nameAdam/v/dense_22/kernel:622
0
_user_specified_nameAdam/m/dense_22/kernel:410
.
_user_specified_nameAdam/v/dense_21/bias:400
.
_user_specified_nameAdam/m/dense_21/bias:6/2
0
_user_specified_nameAdam/v/dense_21/kernel:6.2
0
_user_specified_nameAdam/m/dense_21/kernel:4-0
.
_user_specified_nameAdam/v/dense_20/bias:4,0
.
_user_specified_nameAdam/m/dense_20/bias:6+2
0
_user_specified_nameAdam/v/dense_20/kernel:6*2
0
_user_specified_nameAdam/m/dense_20/kernel:4)0
.
_user_specified_nameAdam/v/dense_19/bias:4(0
.
_user_specified_nameAdam/m/dense_19/bias:6'2
0
_user_specified_nameAdam/v/dense_19/kernel:6&2
0
_user_specified_nameAdam/m/dense_19/kernel:=%9
7
_user_specified_nameAdam/v/lstm_23/lstm_cell/bias:=$9
7
_user_specified_nameAdam/m/lstm_23/lstm_cell/bias:I#E
C
_user_specified_name+)Adam/v/lstm_23/lstm_cell/recurrent_kernel:I"E
C
_user_specified_name+)Adam/m/lstm_23/lstm_cell/recurrent_kernel:?!;
9
_user_specified_name!Adam/v/lstm_23/lstm_cell/kernel:? ;
9
_user_specified_name!Adam/m/lstm_23/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_22/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_22/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_22/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_22/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_22/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_22/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_21/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_21/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_21/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_21/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_21/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_21/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:62
0
_user_specified_namelstm_23/lstm_cell/bias:B>
<
_user_specified_name$"lstm_23/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_23/lstm_cell/kernel:62
0
_user_specified_namelstm_22/lstm_cell/bias:B>
<
_user_specified_name$"lstm_22/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_22/lstm_cell/kernel:62
0
_user_specified_namelstm_21/lstm_cell/bias:B
>
<
_user_specified_name$"lstm_21/lstm_cell/recurrent_kernel:8	4
2
_user_specified_namelstm_21/lstm_cell/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:-)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
щ9
»
while_body_509908
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
└I
є
C__inference_lstm_22_layer_call_and_return_conditional_losses_509634

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509550*
condR
while_cond_509549*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
╝$
╚
while_body_506161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_506185_0:@*
while_lstm_cell_506187_0:@&
while_lstm_cell_506189_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_506185:@(
while_lstm_cell_506187:@$
while_lstm_cell_506189:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_506185_0while_lstm_cell_506187_0while_lstm_cell_506189_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506147┘
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_506185while_lstm_cell_506185_0"2
while_lstm_cell_506187while_lstm_cell_506187_0"2
while_lstm_cell_506189while_lstm_cell_506189_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name506189:&	"
 
_user_specified_name506187:&"
 
_user_specified_name506185:_[
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
:         :-)
'
_output_shapes
:         :
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
while_cond_509049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509049___redundant_placeholder04
0while_while_cond_509049___redundant_placeholder14
0while_while_cond_509049___redundant_placeholder24
0while_while_cond_509049___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
¤
d
+__inference_dropout_37_layer_call_fn_510527

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_507677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
а9
Ш
C__inference_lstm_23_layer_call_and_return_conditional_losses_506924

inputs"
lstm_cell_506840:@"
lstm_cell_506842:@
lstm_cell_506844:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506840lstm_cell_506842lstm_cell_506844*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506839n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506840lstm_cell_506842lstm_cell_506844*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506854*
condR
while_cond_506853*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506844:&"
 
_user_specified_name506842:&"
 
_user_specified_name506840:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
П8
»
while_body_508621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
└I
є
C__inference_lstm_22_layer_call_and_return_conditional_losses_507435

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507351*
condR
while_cond_507350*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
П8
»
while_body_507351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
з
┬
-__inference_sequential_7_layer_call_fn_508314
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_508236o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508310:&"
 
_user_specified_name508308:&"
 
_user_specified_name508306:&"
 
_user_specified_name508304:&"
 
_user_specified_name508302:&"
 
_user_specified_name508300:&"
 
_user_specified_name508298:&
"
 
_user_specified_name508296:&	"
 
_user_specified_name508294:&"
 
_user_specified_name508292:&"
 
_user_specified_name508290:&"
 
_user_specified_name508288:&"
 
_user_specified_name508286:&"
 
_user_specified_name508284:&"
 
_user_specified_name508282:&"
 
_user_specified_name508280:&"
 
_user_specified_name508278:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_21_input
┬	
├
while_cond_508763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_508763___redundant_placeholder04
0while_while_cond_508763___redundant_placeholder14
0while_while_cond_508763___redundant_placeholder24
0while_while_cond_508763___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
жJ
ѕ
C__inference_lstm_23_layer_call_and_return_conditional_losses_509993
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509908*
condR
while_cond_509907*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
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
 :                  
"
_user_specified_name
inputs_0
Щ
┤
(__inference_lstm_21_layer_call_fn_508529
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_506230|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508525:&"
 
_user_specified_name508523:&"
 
_user_specified_name508521:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╚
▓
(__inference_lstm_23_layer_call_fn_509837

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_507600o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509833:&"
 
_user_specified_name509831:&"
 
_user_specified_name509829:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬	
├
while_cond_506160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_506160___redundant_placeholder04
0while_while_cond_506160___redundant_placeholder14
0while_while_cond_506160___redundant_placeholder24
0while_while_cond_506160___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
э	
ш
D__inference_dense_22_layer_call_and_return_conditional_losses_510615

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:         : : 20
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
:         
 
_user_specified_nameinputs
╦

ш
D__inference_dense_20_layer_call_and_return_conditional_losses_510522

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
Я
┤
(__inference_lstm_23_layer_call_fn_509815
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_506924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509811:&"
 
_user_specified_name509809:&"
 
_user_specified_name509807:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
┬	
├
while_cond_507187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507187___redundant_placeholder04
0while_while_cond_507187___redundant_placeholder14
0while_while_cond_507187___redundant_placeholder24
0while_while_cond_507187___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
while_cond_507000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507000___redundant_placeholder04
0while_while_cond_507000___redundant_placeholder14
0while_while_cond_507000___redundant_placeholder24
0while_while_cond_507000___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
╬
у
&sequential_7_lstm_23_while_cond_505968F
Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counterL
Hsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations*
&sequential_7_lstm_23_while_placeholder,
(sequential_7_lstm_23_while_placeholder_1,
(sequential_7_lstm_23_while_placeholder_2,
(sequential_7_lstm_23_while_placeholder_3H
Dsequential_7_lstm_23_while_less_sequential_7_lstm_23_strided_slice_1^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_505968___redundant_placeholder0^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_505968___redundant_placeholder1^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_505968___redundant_placeholder2^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_505968___redundant_placeholder3'
#sequential_7_lstm_23_while_identity
Х
sequential_7/lstm_23/while/LessLess&sequential_7_lstm_23_while_placeholderDsequential_7_lstm_23_while_less_sequential_7_lstm_23_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_23/while/IdentityIdentity#sequential_7/lstm_23/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_23_while_identity,sequential_7/lstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::

_output_shapes
::\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_23/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_23/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_23/while/loop_counter
┘
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_508228

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ж
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_509804

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510811

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
d
+__inference_dropout_38_layer_call_fn_510574

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_507706o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Х
­
*__inference_lstm_cell_layer_call_fn_510730

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510722:&"
 
_user_specified_name510720:&"
 
_user_specified_name510718:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Х
­
*__inference_lstm_cell_layer_call_fn_510649

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510641:&"
 
_user_specified_name510639:&"
 
_user_specified_name510637:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦

ш
D__inference_dense_20_layer_call_and_return_conditional_losses_507660

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
Щ
┤
(__inference_lstm_22_layer_call_fn_509183
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_506721|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509179:&"
 
_user_specified_name509177:&"
 
_user_specified_name509175:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
џ

e
F__inference_dropout_36_layer_call_and_return_conditional_losses_507648

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
кJ
є
C__inference_lstm_23_layer_call_and_return_conditional_losses_510283

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_510198*
condR
while_cond_510197*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
■I
ѕ
C__inference_lstm_22_layer_call_and_return_conditional_losses_509491
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509407*
condR
while_cond_509406*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
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
 :                  
"
_user_specified_name
inputs_0
щ9
»
while_body_510053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
while_cond_506506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_506506___redundant_placeholder04
0while_while_cond_506506___redundant_placeholder14
0while_while_cond_506506___redundant_placeholder24
0while_while_cond_506506___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
╝$
╚
while_body_506652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_506676_0:@*
while_lstm_cell_506678_0:@&
while_lstm_cell_506680_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_506676:@(
while_lstm_cell_506678:@$
while_lstm_cell_506680:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_506676_0while_lstm_cell_506678_0while_lstm_cell_506680_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506638┘
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_506676while_lstm_cell_506676_0"2
while_lstm_cell_506678while_lstm_cell_506678_0"2
while_lstm_cell_506680while_lstm_cell_506680_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name506680:&	"
 
_user_specified_name506678:&"
 
_user_specified_name506676:_[
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_33_layer_call_and_return_conditional_losses_509161

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
кJ
є
C__inference_lstm_23_layer_call_and_return_conditional_losses_507600

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507515*
condR
while_cond_507514*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
Щ
┤
(__inference_lstm_21_layer_call_fn_508540
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_506375|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508536:&"
 
_user_specified_name508534:&"
 
_user_specified_name508532:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
х8
Ш
C__inference_lstm_22_layer_call_and_return_conditional_losses_506576

inputs"
lstm_cell_506494:@"
lstm_cell_506496:@
lstm_cell_506498:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506494lstm_cell_506496lstm_cell_506498*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506493n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506494lstm_cell_506496lstm_cell_506498*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506507*
condR
while_cond_506506*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506498:&"
 
_user_specified_name506496:&"
 
_user_specified_name506494:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
л
▓
(__inference_lstm_21_layer_call_fn_508562

inputs
unknown:@
	unknown_0:@
	unknown_1:@
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
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_507869s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508558:&"
 
_user_specified_name508556:&"
 
_user_specified_name508554:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506147

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь
ќ
)__inference_dense_22_layer_call_fn_510605

inputs
unknown:
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
D__inference_dense_22_layer_call_and_return_conditional_losses_507717o
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
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510601:&"
 
_user_specified_name510599:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ

e
F__inference_dropout_36_layer_call_and_return_conditional_losses_510497

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
А
G
+__inference_dropout_36_layer_call_fn_510485

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_508206`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_508195

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
кJ
є
C__inference_lstm_23_layer_call_and_return_conditional_losses_510428

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_510343*
condR
while_cond_510342*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
п%
╚
while_body_506854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_506878_0:@*
while_lstm_cell_506880_0:@&
while_lstm_cell_506882_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_506878:@(
while_lstm_cell_506880:@$
while_lstm_cell_506882:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_506878_0while_lstm_cell_506880_0while_lstm_cell_506882_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506839r
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_506878while_lstm_cell_506878_0"2
while_lstm_cell_506880while_lstm_cell_506880_0"2
while_lstm_cell_506882while_lstm_cell_506882_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name506882:&	"
 
_user_specified_name506880:&"
 
_user_specified_name506878:_[
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
:         :-)
'
_output_shapes
:         :
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
while_cond_510342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_510342___redundant_placeholder04
0while_while_cond_510342___redundant_placeholder14
0while_while_cond_510342___redundant_placeholder24
0while_while_cond_510342___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
while_cond_507784
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507784___redundant_placeholder04
0while_while_cond_507784___redundant_placeholder14
0while_while_cond_507784___redundant_placeholder24
0while_while_cond_507784___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
├
╣
$__inference_signature_wrapper_508518
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_506085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508514:&"
 
_user_specified_name508512:&"
 
_user_specified_name508510:&"
 
_user_specified_name508508:&"
 
_user_specified_name508506:&"
 
_user_specified_name508504:&"
 
_user_specified_name508502:&
"
 
_user_specified_name508500:&	"
 
_user_specified_name508498:&"
 
_user_specified_name508496:&"
 
_user_specified_name508494:&"
 
_user_specified_name508492:&"
 
_user_specified_name508490:&"
 
_user_specified_name508488:&"
 
_user_specified_name508486:&"
 
_user_specified_name508484:&"
 
_user_specified_name508482:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_21_input
┘
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_508206

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ж
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_507881

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬	
├
while_cond_510197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_510197___redundant_placeholder04
0while_while_cond_510197___redundant_placeholder14
0while_while_cond_510197___redundant_placeholder24
0while_while_cond_510197___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_38_layer_call_and_return_conditional_losses_510596

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
■I
ѕ
C__inference_lstm_21_layer_call_and_return_conditional_losses_508705
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508621*
condR
while_cond_508620*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
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
П8
»
while_body_509693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
└I
є
C__inference_lstm_21_layer_call_and_return_conditional_losses_507272

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507188*
condR
while_cond_507187*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
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
¤
d
+__inference_dropout_36_layer_call_fn_510480

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_507648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_508906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_508906___redundant_placeholder04
0while_while_cond_508906___redundant_placeholder14
0while_while_cond_508906___redundant_placeholder24
0while_while_cond_508906___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_510455

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
љR
л
&sequential_7_lstm_21_while_body_505688F
Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counterL
Hsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations*
&sequential_7_lstm_21_while_placeholder,
(sequential_7_lstm_21_while_placeholder_1,
(sequential_7_lstm_21_while_placeholder_2,
(sequential_7_lstm_21_while_placeholder_3E
Asequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1_0Ђ
}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0W
Esequential_7_lstm_21_while_lstm_cell_matmul_readvariableop_resource_0:@Y
Gsequential_7_lstm_21_while_lstm_cell_matmul_1_readvariableop_resource_0:@T
Fsequential_7_lstm_21_while_lstm_cell_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_21_while_identity)
%sequential_7_lstm_21_while_identity_1)
%sequential_7_lstm_21_while_identity_2)
%sequential_7_lstm_21_while_identity_3)
%sequential_7_lstm_21_while_identity_4)
%sequential_7_lstm_21_while_identity_5C
?sequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1
{sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensorU
Csequential_7_lstm_21_while_lstm_cell_matmul_readvariableop_resource:@W
Esequential_7_lstm_21_while_lstm_cell_matmul_1_readvariableop_resource:@R
Dsequential_7_lstm_21_while_lstm_cell_biasadd_readvariableop_resource:@ѕб;sequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOpб:sequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOpб<sequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_21_while_placeholderUsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0└
:sequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpEsequential_7_lstm_21_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ы
+sequential_7/lstm_21/while/lstm_cell/MatMulMatMulEsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @─
<sequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpGsequential_7_lstm_21_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0┘
-sequential_7/lstm_21/while/lstm_cell/MatMul_1MatMul(sequential_7_lstm_21_while_placeholder_2Dsequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @М
(sequential_7/lstm_21/while/lstm_cell/addAddV25sequential_7/lstm_21/while/lstm_cell/MatMul:product:07sequential_7/lstm_21/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @Й
;sequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpFsequential_7_lstm_21_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0▄
,sequential_7/lstm_21/while/lstm_cell/BiasAddBiasAdd,sequential_7/lstm_21/while/lstm_cell/add:z:0Csequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
4sequential_7/lstm_21/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_7/lstm_21/while/lstm_cell/splitSplit=sequential_7/lstm_21/while/lstm_cell/split/split_dim:output:05sequential_7/lstm_21/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitъ
,sequential_7/lstm_21/while/lstm_cell/SigmoidSigmoid3sequential_7/lstm_21/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_21/while/lstm_cell/Sigmoid_1Sigmoid3sequential_7/lstm_21/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ┐
(sequential_7/lstm_21/while/lstm_cell/mulMul2sequential_7/lstm_21/while/lstm_cell/Sigmoid_1:y:0(sequential_7_lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         ў
)sequential_7/lstm_21/while/lstm_cell/ReluRelu3sequential_7/lstm_21/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╬
*sequential_7/lstm_21/while/lstm_cell/mul_1Mul0sequential_7/lstm_21/while/lstm_cell/Sigmoid:y:07sequential_7/lstm_21/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ├
*sequential_7/lstm_21/while/lstm_cell/add_1AddV2,sequential_7/lstm_21/while/lstm_cell/mul:z:0.sequential_7/lstm_21/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_21/while/lstm_cell/Sigmoid_2Sigmoid3sequential_7/lstm_21/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ћ
+sequential_7/lstm_21/while/lstm_cell/Relu_1Relu.sequential_7/lstm_21/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         м
*sequential_7/lstm_21/while/lstm_cell/mul_2Mul2sequential_7/lstm_21/while/lstm_cell/Sigmoid_2:y:09sequential_7/lstm_21/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ќ
?sequential_7/lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_21_while_placeholder_1&sequential_7_lstm_21_while_placeholder.sequential_7/lstm_21/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_21/while/addAddV2&sequential_7_lstm_21_while_placeholder)sequential_7/lstm_21/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_21/while/add_1AddV2Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counter+sequential_7/lstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_21/while/IdentityIdentity$sequential_7/lstm_21/while/add_1:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_21/while/Identity_1IdentityHsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_21/while/Identity_2Identity"sequential_7/lstm_21/while/add:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_21/while/Identity_3IdentityOsequential_7/lstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: х
%sequential_7/lstm_21/while/Identity_4Identity.sequential_7/lstm_21/while/lstm_cell/mul_2:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*'
_output_shapes
:         х
%sequential_7/lstm_21/while/Identity_5Identity.sequential_7/lstm_21/while/lstm_cell/add_1:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*'
_output_shapes
:         э
sequential_7/lstm_21/while/NoOpNoOp<^sequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOp;^sequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOp=^sequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "W
%sequential_7_lstm_21_while_identity_1.sequential_7/lstm_21/while/Identity_1:output:0"W
%sequential_7_lstm_21_while_identity_2.sequential_7/lstm_21/while/Identity_2:output:0"W
%sequential_7_lstm_21_while_identity_3.sequential_7/lstm_21/while/Identity_3:output:0"W
%sequential_7_lstm_21_while_identity_4.sequential_7/lstm_21/while/Identity_4:output:0"W
%sequential_7_lstm_21_while_identity_5.sequential_7/lstm_21/while/Identity_5:output:0"S
#sequential_7_lstm_21_while_identity,sequential_7/lstm_21/while/Identity:output:0"ј
Dsequential_7_lstm_21_while_lstm_cell_biasadd_readvariableop_resourceFsequential_7_lstm_21_while_lstm_cell_biasadd_readvariableop_resource_0"љ
Esequential_7_lstm_21_while_lstm_cell_matmul_1_readvariableop_resourceGsequential_7_lstm_21_while_lstm_cell_matmul_1_readvariableop_resource_0"ї
Csequential_7_lstm_21_while_lstm_cell_matmul_readvariableop_resourceEsequential_7_lstm_21_while_lstm_cell_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1Asequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1_0"Ч
{sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2z
;sequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOp;sequential_7/lstm_21/while/lstm_cell/BiasAdd/ReadVariableOp2x
:sequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOp:sequential_7/lstm_21/while/lstm_cell/MatMul/ReadVariableOp2|
<sequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOp<sequential_7/lstm_21/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:tp

_output_shapes
: 
V
_user_specified_name><sequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor:\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_21/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_21/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_21/while/loop_counter
П8
»
while_body_507941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506638

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь
ќ
)__inference_dense_20_layer_call_fn_510511

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_507660o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510507:&"
 
_user_specified_name510505:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х8
Ш
C__inference_lstm_21_layer_call_and_return_conditional_losses_506375

inputs"
lstm_cell_506293:@"
lstm_cell_506295:@
lstm_cell_506297:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506293lstm_cell_506295lstm_cell_506297*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506292n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506293lstm_cell_506295lstm_cell_506297*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506306*
condR
while_cond_506305*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506297:&"
 
_user_specified_name506295:&"
 
_user_specified_name506293:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
џ

e
F__inference_dropout_35_layer_call_and_return_conditional_losses_510450

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506292

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╬
у
&sequential_7_lstm_22_while_cond_505827F
Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counterL
Hsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations*
&sequential_7_lstm_22_while_placeholder,
(sequential_7_lstm_22_while_placeholder_1,
(sequential_7_lstm_22_while_placeholder_2,
(sequential_7_lstm_22_while_placeholder_3H
Dsequential_7_lstm_22_while_less_sequential_7_lstm_22_strided_slice_1^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_505827___redundant_placeholder0^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_505827___redundant_placeholder1^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_505827___redundant_placeholder2^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_505827___redundant_placeholder3'
#sequential_7_lstm_22_while_identity
Х
sequential_7/lstm_22/while/LessLess&sequential_7_lstm_22_while_placeholderDsequential_7_lstm_22_while_less_sequential_7_lstm_22_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_22/while/IdentityIdentity#sequential_7/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_22_while_identity,sequential_7/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::

_output_shapes
::\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_22/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_22/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_22/while/loop_counter
Х
­
*__inference_lstm_cell_layer_call_fn_510845

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506986o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510837:&"
 
_user_specified_name510835:&"
 
_user_specified_name510833:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀
d
+__inference_dropout_34_layer_call_fn_509782

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
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_507454s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Х

e
F__inference_dropout_33_layer_call_and_return_conditional_losses_509156

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
Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
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
T
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
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬	
├
while_cond_507350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507350___redundant_placeholder04
0while_while_cond_507350___redundant_placeholder14
0while_while_cond_507350___redundant_placeholder24
0while_while_cond_507350___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
По
ћ
!__inference__wrapped_model_506085
lstm_21_inputO
=sequential_7_lstm_21_lstm_cell_matmul_readvariableop_resource:@Q
?sequential_7_lstm_21_lstm_cell_matmul_1_readvariableop_resource:@L
>sequential_7_lstm_21_lstm_cell_biasadd_readvariableop_resource:@O
=sequential_7_lstm_22_lstm_cell_matmul_readvariableop_resource:@Q
?sequential_7_lstm_22_lstm_cell_matmul_1_readvariableop_resource:@L
>sequential_7_lstm_22_lstm_cell_biasadd_readvariableop_resource:@O
=sequential_7_lstm_23_lstm_cell_matmul_readvariableop_resource:@Q
?sequential_7_lstm_23_lstm_cell_matmul_1_readvariableop_resource:@L
>sequential_7_lstm_23_lstm_cell_biasadd_readvariableop_resource:@F
4sequential_7_dense_19_matmul_readvariableop_resource:C
5sequential_7_dense_19_biasadd_readvariableop_resource:F
4sequential_7_dense_20_matmul_readvariableop_resource:C
5sequential_7_dense_20_biasadd_readvariableop_resource:F
4sequential_7_dense_21_matmul_readvariableop_resource:C
5sequential_7_dense_21_biasadd_readvariableop_resource:F
4sequential_7_dense_22_matmul_readvariableop_resource:C
5sequential_7_dense_22_biasadd_readvariableop_resource:
identityѕб,sequential_7/dense_19/BiasAdd/ReadVariableOpб+sequential_7/dense_19/MatMul/ReadVariableOpб,sequential_7/dense_20/BiasAdd/ReadVariableOpб+sequential_7/dense_20/MatMul/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOpб5sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOpб4sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOpб6sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOpбsequential_7/lstm_21/whileб5sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOpб4sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOpб6sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOpбsequential_7/lstm_22/whileб5sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOpб4sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOpб6sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOpбsequential_7/lstm_23/whilee
sequential_7/lstm_21/ShapeShapelstm_21_input*
T0*
_output_shapes
::ь¤r
(sequential_7/lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_21/strided_sliceStridedSlice#sequential_7/lstm_21/Shape:output:01sequential_7/lstm_21/strided_slice/stack:output:03sequential_7/lstm_21/strided_slice/stack_1:output:03sequential_7/lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_21/zeros/packedPack+sequential_7/lstm_21/strided_slice:output:0,sequential_7/lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_21/zerosFill*sequential_7/lstm_21/zeros/packed:output:0)sequential_7/lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_21/zeros_1/packedPack+sequential_7/lstm_21/strided_slice:output:0.sequential_7/lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_21/zeros_1Fill,sequential_7/lstm_21/zeros_1/packed:output:0+sequential_7/lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ъ
sequential_7/lstm_21/transpose	Transposelstm_21_input,sequential_7/lstm_21/transpose/perm:output:0*
T0*+
_output_shapes
:
         |
sequential_7/lstm_21/Shape_1Shape"sequential_7/lstm_21/transpose:y:0*
T0*
_output_shapes
::ь¤t
*sequential_7/lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_21/strided_slice_1StridedSlice%sequential_7/lstm_21/Shape_1:output:03sequential_7/lstm_21/strided_slice_1/stack:output:05sequential_7/lstm_21/strided_slice_1/stack_1:output:05sequential_7/lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_21/TensorArrayV2TensorListReserve9sequential_7/lstm_21/TensorArrayV2/element_shape:output:0-sequential_7/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_21/transpose:y:0Ssequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_21/strided_slice_2StridedSlice"sequential_7/lstm_21/transpose:y:03sequential_7/lstm_21/strided_slice_2/stack:output:05sequential_7/lstm_21/strided_slice_2/stack_1:output:05sequential_7/lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask▓
4sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOpReadVariableOp=sequential_7_lstm_21_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╬
%sequential_7/lstm_21/lstm_cell/MatMulMatMul-sequential_7/lstm_21/strided_slice_2:output:0<sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Х
6sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?sequential_7_lstm_21_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╚
'sequential_7/lstm_21/lstm_cell/MatMul_1MatMul#sequential_7/lstm_21/zeros:output:0>sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┴
"sequential_7/lstm_21/lstm_cell/addAddV2/sequential_7/lstm_21/lstm_cell/MatMul:product:01sequential_7/lstm_21/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @░
5sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>sequential_7_lstm_21_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╩
&sequential_7/lstm_21/lstm_cell/BiasAddBiasAdd&sequential_7/lstm_21/lstm_cell/add:z:0=sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
.sequential_7/lstm_21/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
$sequential_7/lstm_21/lstm_cell/splitSplit7sequential_7/lstm_21/lstm_cell/split/split_dim:output:0/sequential_7/lstm_21/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitњ
&sequential_7/lstm_21/lstm_cell/SigmoidSigmoid-sequential_7/lstm_21/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_21/lstm_cell/Sigmoid_1Sigmoid-sequential_7/lstm_21/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ░
"sequential_7/lstm_21/lstm_cell/mulMul,sequential_7/lstm_21/lstm_cell/Sigmoid_1:y:0%sequential_7/lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         ї
#sequential_7/lstm_21/lstm_cell/ReluRelu-sequential_7/lstm_21/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╝
$sequential_7/lstm_21/lstm_cell/mul_1Mul*sequential_7/lstm_21/lstm_cell/Sigmoid:y:01sequential_7/lstm_21/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ▒
$sequential_7/lstm_21/lstm_cell/add_1AddV2&sequential_7/lstm_21/lstm_cell/mul:z:0(sequential_7/lstm_21/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_21/lstm_cell/Sigmoid_2Sigmoid-sequential_7/lstm_21/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ѕ
%sequential_7/lstm_21/lstm_cell/Relu_1Relu(sequential_7/lstm_21/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         └
$sequential_7/lstm_21/lstm_cell/mul_2Mul,sequential_7/lstm_21/lstm_cell/Sigmoid_2:y:03sequential_7/lstm_21/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       э
$sequential_7/lstm_21/TensorArrayV2_1TensorListReserve;sequential_7/lstm_21/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
sequential_7/lstm_21/whileWhile0sequential_7/lstm_21/while/loop_counter:output:06sequential_7/lstm_21/while/maximum_iterations:output:0"sequential_7/lstm_21/time:output:0-sequential_7/lstm_21/TensorArrayV2_1:handle:0#sequential_7/lstm_21/zeros:output:0%sequential_7/lstm_21/zeros_1:output:0-sequential_7/lstm_21/strided_slice_1:output:0Lsequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_7_lstm_21_lstm_cell_matmul_readvariableop_resource?sequential_7_lstm_21_lstm_cell_matmul_1_readvariableop_resource>sequential_7_lstm_21_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*2
body*R(
&sequential_7_lstm_21_while_body_505688*2
cond*R(
&sequential_7_lstm_21_while_cond_505687*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
7sequential_7/lstm_21/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_21/while:output:3Nsequential_7/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
element_dtype0}
*sequential_7/lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_21/strided_slice_3StridedSlice@sequential_7/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_21/strided_slice_3/stack:output:05sequential_7/lstm_21/strided_slice_3/stack_1:output:05sequential_7/lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_21/transpose_1	Transpose@sequential_7/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
p
sequential_7/lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ѕ
 sequential_7/dropout_33/IdentityIdentity$sequential_7/lstm_21/transpose_1:y:0*
T0*+
_output_shapes
:         
Ђ
sequential_7/lstm_22/ShapeShape)sequential_7/dropout_33/Identity:output:0*
T0*
_output_shapes
::ь¤r
(sequential_7/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_22/strided_sliceStridedSlice#sequential_7/lstm_22/Shape:output:01sequential_7/lstm_22/strided_slice/stack:output:03sequential_7/lstm_22/strided_slice/stack_1:output:03sequential_7/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_22/zeros/packedPack+sequential_7/lstm_22/strided_slice:output:0,sequential_7/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_22/zerosFill*sequential_7/lstm_22/zeros/packed:output:0)sequential_7/lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_22/zeros_1/packedPack+sequential_7/lstm_22/strided_slice:output:0.sequential_7/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_22/zeros_1Fill,sequential_7/lstm_22/zeros_1/packed:output:0+sequential_7/lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ║
sequential_7/lstm_22/transpose	Transpose)sequential_7/dropout_33/Identity:output:0,sequential_7/lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:
         |
sequential_7/lstm_22/Shape_1Shape"sequential_7/lstm_22/transpose:y:0*
T0*
_output_shapes
::ь¤t
*sequential_7/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_22/strided_slice_1StridedSlice%sequential_7/lstm_22/Shape_1:output:03sequential_7/lstm_22/strided_slice_1/stack:output:05sequential_7/lstm_22/strided_slice_1/stack_1:output:05sequential_7/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_22/TensorArrayV2TensorListReserve9sequential_7/lstm_22/TensorArrayV2/element_shape:output:0-sequential_7/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_22/transpose:y:0Ssequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_22/strided_slice_2StridedSlice"sequential_7/lstm_22/transpose:y:03sequential_7/lstm_22/strided_slice_2/stack:output:05sequential_7/lstm_22/strided_slice_2/stack_1:output:05sequential_7/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask▓
4sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOpReadVariableOp=sequential_7_lstm_22_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╬
%sequential_7/lstm_22/lstm_cell/MatMulMatMul-sequential_7/lstm_22/strided_slice_2:output:0<sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Х
6sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?sequential_7_lstm_22_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╚
'sequential_7/lstm_22/lstm_cell/MatMul_1MatMul#sequential_7/lstm_22/zeros:output:0>sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┴
"sequential_7/lstm_22/lstm_cell/addAddV2/sequential_7/lstm_22/lstm_cell/MatMul:product:01sequential_7/lstm_22/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @░
5sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>sequential_7_lstm_22_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╩
&sequential_7/lstm_22/lstm_cell/BiasAddBiasAdd&sequential_7/lstm_22/lstm_cell/add:z:0=sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
.sequential_7/lstm_22/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
$sequential_7/lstm_22/lstm_cell/splitSplit7sequential_7/lstm_22/lstm_cell/split/split_dim:output:0/sequential_7/lstm_22/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitњ
&sequential_7/lstm_22/lstm_cell/SigmoidSigmoid-sequential_7/lstm_22/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_22/lstm_cell/Sigmoid_1Sigmoid-sequential_7/lstm_22/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ░
"sequential_7/lstm_22/lstm_cell/mulMul,sequential_7/lstm_22/lstm_cell/Sigmoid_1:y:0%sequential_7/lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         ї
#sequential_7/lstm_22/lstm_cell/ReluRelu-sequential_7/lstm_22/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╝
$sequential_7/lstm_22/lstm_cell/mul_1Mul*sequential_7/lstm_22/lstm_cell/Sigmoid:y:01sequential_7/lstm_22/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ▒
$sequential_7/lstm_22/lstm_cell/add_1AddV2&sequential_7/lstm_22/lstm_cell/mul:z:0(sequential_7/lstm_22/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_22/lstm_cell/Sigmoid_2Sigmoid-sequential_7/lstm_22/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ѕ
%sequential_7/lstm_22/lstm_cell/Relu_1Relu(sequential_7/lstm_22/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         └
$sequential_7/lstm_22/lstm_cell/mul_2Mul,sequential_7/lstm_22/lstm_cell/Sigmoid_2:y:03sequential_7/lstm_22/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       э
$sequential_7/lstm_22/TensorArrayV2_1TensorListReserve;sequential_7/lstm_22/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
sequential_7/lstm_22/whileWhile0sequential_7/lstm_22/while/loop_counter:output:06sequential_7/lstm_22/while/maximum_iterations:output:0"sequential_7/lstm_22/time:output:0-sequential_7/lstm_22/TensorArrayV2_1:handle:0#sequential_7/lstm_22/zeros:output:0%sequential_7/lstm_22/zeros_1:output:0-sequential_7/lstm_22/strided_slice_1:output:0Lsequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_7_lstm_22_lstm_cell_matmul_readvariableop_resource?sequential_7_lstm_22_lstm_cell_matmul_1_readvariableop_resource>sequential_7_lstm_22_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*2
body*R(
&sequential_7_lstm_22_while_body_505828*2
cond*R(
&sequential_7_lstm_22_while_cond_505827*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
7sequential_7/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_22/while:output:3Nsequential_7/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
element_dtype0}
*sequential_7/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_22/strided_slice_3StridedSlice@sequential_7/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_22/strided_slice_3/stack:output:05sequential_7/lstm_22/strided_slice_3/stack_1:output:05sequential_7/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_22/transpose_1	Transpose@sequential_7/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
p
sequential_7/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ѕ
 sequential_7/dropout_34/IdentityIdentity$sequential_7/lstm_22/transpose_1:y:0*
T0*+
_output_shapes
:         
Ђ
sequential_7/lstm_23/ShapeShape)sequential_7/dropout_34/Identity:output:0*
T0*
_output_shapes
::ь¤r
(sequential_7/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_23/strided_sliceStridedSlice#sequential_7/lstm_23/Shape:output:01sequential_7/lstm_23/strided_slice/stack:output:03sequential_7/lstm_23/strided_slice/stack_1:output:03sequential_7/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_23/zeros/packedPack+sequential_7/lstm_23/strided_slice:output:0,sequential_7/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_23/zerosFill*sequential_7/lstm_23/zeros/packed:output:0)sequential_7/lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_23/zeros_1/packedPack+sequential_7/lstm_23/strided_slice:output:0.sequential_7/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_23/zeros_1Fill,sequential_7/lstm_23/zeros_1/packed:output:0+sequential_7/lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ║
sequential_7/lstm_23/transpose	Transpose)sequential_7/dropout_34/Identity:output:0,sequential_7/lstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:
         |
sequential_7/lstm_23/Shape_1Shape"sequential_7/lstm_23/transpose:y:0*
T0*
_output_shapes
::ь¤t
*sequential_7/lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_23/strided_slice_1StridedSlice%sequential_7/lstm_23/Shape_1:output:03sequential_7/lstm_23/strided_slice_1/stack:output:05sequential_7/lstm_23/strided_slice_1/stack_1:output:05sequential_7/lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_23/TensorArrayV2TensorListReserve9sequential_7/lstm_23/TensorArrayV2/element_shape:output:0-sequential_7/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_23/transpose:y:0Ssequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_23/strided_slice_2StridedSlice"sequential_7/lstm_23/transpose:y:03sequential_7/lstm_23/strided_slice_2/stack:output:05sequential_7/lstm_23/strided_slice_2/stack_1:output:05sequential_7/lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask▓
4sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOpReadVariableOp=sequential_7_lstm_23_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╬
%sequential_7/lstm_23/lstm_cell/MatMulMatMul-sequential_7/lstm_23/strided_slice_2:output:0<sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Х
6sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?sequential_7_lstm_23_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╚
'sequential_7/lstm_23/lstm_cell/MatMul_1MatMul#sequential_7/lstm_23/zeros:output:0>sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┴
"sequential_7/lstm_23/lstm_cell/addAddV2/sequential_7/lstm_23/lstm_cell/MatMul:product:01sequential_7/lstm_23/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @░
5sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>sequential_7_lstm_23_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╩
&sequential_7/lstm_23/lstm_cell/BiasAddBiasAdd&sequential_7/lstm_23/lstm_cell/add:z:0=sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
.sequential_7/lstm_23/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Њ
$sequential_7/lstm_23/lstm_cell/splitSplit7sequential_7/lstm_23/lstm_cell/split/split_dim:output:0/sequential_7/lstm_23/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitњ
&sequential_7/lstm_23/lstm_cell/SigmoidSigmoid-sequential_7/lstm_23/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_23/lstm_cell/Sigmoid_1Sigmoid-sequential_7/lstm_23/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ░
"sequential_7/lstm_23/lstm_cell/mulMul,sequential_7/lstm_23/lstm_cell/Sigmoid_1:y:0%sequential_7/lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         ї
#sequential_7/lstm_23/lstm_cell/ReluRelu-sequential_7/lstm_23/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╝
$sequential_7/lstm_23/lstm_cell/mul_1Mul*sequential_7/lstm_23/lstm_cell/Sigmoid:y:01sequential_7/lstm_23/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ▒
$sequential_7/lstm_23/lstm_cell/add_1AddV2&sequential_7/lstm_23/lstm_cell/mul:z:0(sequential_7/lstm_23/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ћ
(sequential_7/lstm_23/lstm_cell/Sigmoid_2Sigmoid-sequential_7/lstm_23/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ѕ
%sequential_7/lstm_23/lstm_cell/Relu_1Relu(sequential_7/lstm_23/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         └
$sequential_7/lstm_23/lstm_cell/mul_2Mul,sequential_7/lstm_23/lstm_cell/Sigmoid_2:y:03sequential_7/lstm_23/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
1sequential_7/lstm_23/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ё
$sequential_7/lstm_23/TensorArrayV2_1TensorListReserve;sequential_7/lstm_23/TensorArrayV2_1/element_shape:output:0:sequential_7/lstm_23/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
sequential_7/lstm_23/whileWhile0sequential_7/lstm_23/while/loop_counter:output:06sequential_7/lstm_23/while/maximum_iterations:output:0"sequential_7/lstm_23/time:output:0-sequential_7/lstm_23/TensorArrayV2_1:handle:0#sequential_7/lstm_23/zeros:output:0%sequential_7/lstm_23/zeros_1:output:0-sequential_7/lstm_23/strided_slice_1:output:0Lsequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_7_lstm_23_lstm_cell_matmul_readvariableop_resource?sequential_7_lstm_23_lstm_cell_matmul_1_readvariableop_resource>sequential_7_lstm_23_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*2
body*R(
&sequential_7_lstm_23_while_body_505969*2
cond*R(
&sequential_7_lstm_23_while_cond_505968*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ћ
7sequential_7/lstm_23/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_23/while:output:3Nsequential_7/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0*
num_elements}
*sequential_7/lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_23/strided_slice_3StridedSlice@sequential_7/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_23/strided_slice_3/stack:output:05sequential_7/lstm_23/strided_slice_3/stack_1:output:05sequential_7/lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_23/transpose_1	Transpose@sequential_7/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         p
sequential_7/lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ї
 sequential_7/dropout_35/IdentityIdentity-sequential_7/lstm_23/strided_slice_3:output:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_19/MatMulMatMul)sequential_7/dropout_35/Identity:output:03sequential_7/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_19/BiasAddBiasAdd&sequential_7/dense_19/MatMul:product:04sequential_7/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_19/ReluRelu&sequential_7/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_36/IdentityIdentity(sequential_7/dense_19/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_20/MatMulMatMul)sequential_7/dropout_36/Identity:output:03sequential_7/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_20/BiasAddBiasAdd&sequential_7/dense_20/MatMul:product:04sequential_7/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_20/ReluRelu&sequential_7/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_37/IdentityIdentity(sequential_7/dense_20/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_21/MatMulMatMul)sequential_7/dropout_37/Identity:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_38/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_38/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_7/dense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         т
NoOpNoOp-^sequential_7/dense_19/BiasAdd/ReadVariableOp,^sequential_7/dense_19/MatMul/ReadVariableOp-^sequential_7/dense_20/BiasAdd/ReadVariableOp,^sequential_7/dense_20/MatMul/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp6^sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOp5^sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOp7^sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOp^sequential_7/lstm_21/while6^sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOp5^sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOp7^sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOp^sequential_7/lstm_22/while6^sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOp5^sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOp7^sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOp^sequential_7/lstm_23/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 2\
,sequential_7/dense_19/BiasAdd/ReadVariableOp,sequential_7/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_19/MatMul/ReadVariableOp+sequential_7/dense_19/MatMul/ReadVariableOp2\
,sequential_7/dense_20/BiasAdd/ReadVariableOp,sequential_7/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_20/MatMul/ReadVariableOp+sequential_7/dense_20/MatMul/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp2n
5sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOp5sequential_7/lstm_21/lstm_cell/BiasAdd/ReadVariableOp2l
4sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOp4sequential_7/lstm_21/lstm_cell/MatMul/ReadVariableOp2p
6sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOp6sequential_7/lstm_21/lstm_cell/MatMul_1/ReadVariableOp28
sequential_7/lstm_21/whilesequential_7/lstm_21/while2n
5sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOp5sequential_7/lstm_22/lstm_cell/BiasAdd/ReadVariableOp2l
4sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOp4sequential_7/lstm_22/lstm_cell/MatMul/ReadVariableOp2p
6sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOp6sequential_7/lstm_22/lstm_cell/MatMul_1/ReadVariableOp28
sequential_7/lstm_22/whilesequential_7/lstm_22/while2n
5sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOp5sequential_7/lstm_23/lstm_cell/BiasAdd/ReadVariableOp2l
4sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOp4sequential_7/lstm_23/lstm_cell/MatMul/ReadVariableOp2p
6sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOp6sequential_7/lstm_23/lstm_cell/MatMul_1/ReadVariableOp28
sequential_7/lstm_23/whilesequential_7/lstm_23/while:($
"
_user_specified_name
resource:($
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
_user_specified_namelstm_21_input
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506839

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
т┤
Ї6
__inference__traced_save_511273
file_prefix8
&read_disablecopyonread_dense_19_kernel:4
&read_1_disablecopyonread_dense_19_bias::
(read_2_disablecopyonread_dense_20_kernel:4
&read_3_disablecopyonread_dense_20_bias::
(read_4_disablecopyonread_dense_21_kernel:4
&read_5_disablecopyonread_dense_21_bias::
(read_6_disablecopyonread_dense_22_kernel:4
&read_7_disablecopyonread_dense_22_bias:C
1read_8_disablecopyonread_lstm_21_lstm_cell_kernel:@M
;read_9_disablecopyonread_lstm_21_lstm_cell_recurrent_kernel:@>
0read_10_disablecopyonread_lstm_21_lstm_cell_bias:@D
2read_11_disablecopyonread_lstm_22_lstm_cell_kernel:@N
<read_12_disablecopyonread_lstm_22_lstm_cell_recurrent_kernel:@>
0read_13_disablecopyonread_lstm_22_lstm_cell_bias:@D
2read_14_disablecopyonread_lstm_23_lstm_cell_kernel:@N
<read_15_disablecopyonread_lstm_23_lstm_cell_recurrent_kernel:@>
0read_16_disablecopyonread_lstm_23_lstm_cell_bias:@-
#read_17_disablecopyonread_iteration:	 1
'read_18_disablecopyonread_learning_rate: K
9read_19_disablecopyonread_adam_m_lstm_21_lstm_cell_kernel:@K
9read_20_disablecopyonread_adam_v_lstm_21_lstm_cell_kernel:@U
Cread_21_disablecopyonread_adam_m_lstm_21_lstm_cell_recurrent_kernel:@U
Cread_22_disablecopyonread_adam_v_lstm_21_lstm_cell_recurrent_kernel:@E
7read_23_disablecopyonread_adam_m_lstm_21_lstm_cell_bias:@E
7read_24_disablecopyonread_adam_v_lstm_21_lstm_cell_bias:@K
9read_25_disablecopyonread_adam_m_lstm_22_lstm_cell_kernel:@K
9read_26_disablecopyonread_adam_v_lstm_22_lstm_cell_kernel:@U
Cread_27_disablecopyonread_adam_m_lstm_22_lstm_cell_recurrent_kernel:@U
Cread_28_disablecopyonread_adam_v_lstm_22_lstm_cell_recurrent_kernel:@E
7read_29_disablecopyonread_adam_m_lstm_22_lstm_cell_bias:@E
7read_30_disablecopyonread_adam_v_lstm_22_lstm_cell_bias:@K
9read_31_disablecopyonread_adam_m_lstm_23_lstm_cell_kernel:@K
9read_32_disablecopyonread_adam_v_lstm_23_lstm_cell_kernel:@U
Cread_33_disablecopyonread_adam_m_lstm_23_lstm_cell_recurrent_kernel:@U
Cread_34_disablecopyonread_adam_v_lstm_23_lstm_cell_recurrent_kernel:@E
7read_35_disablecopyonread_adam_m_lstm_23_lstm_cell_bias:@E
7read_36_disablecopyonread_adam_v_lstm_23_lstm_cell_bias:@B
0read_37_disablecopyonread_adam_m_dense_19_kernel:B
0read_38_disablecopyonread_adam_v_dense_19_kernel:<
.read_39_disablecopyonread_adam_m_dense_19_bias:<
.read_40_disablecopyonread_adam_v_dense_19_bias:B
0read_41_disablecopyonread_adam_m_dense_20_kernel:B
0read_42_disablecopyonread_adam_v_dense_20_kernel:<
.read_43_disablecopyonread_adam_m_dense_20_bias:<
.read_44_disablecopyonread_adam_v_dense_20_bias:B
0read_45_disablecopyonread_adam_m_dense_21_kernel:B
0read_46_disablecopyonread_adam_v_dense_21_kernel:<
.read_47_disablecopyonread_adam_m_dense_21_bias:<
.read_48_disablecopyonread_adam_v_dense_21_bias:B
0read_49_disablecopyonread_adam_m_dense_22_kernel:B
0read_50_disablecopyonread_adam_v_dense_22_kernel:<
.read_51_disablecopyonread_adam_m_dense_22_bias:<
.read_52_disablecopyonread_adam_v_dense_22_bias:+
!read_53_disablecopyonread_total_1: +
!read_54_disablecopyonread_count_1: )
read_55_disablecopyonread_total: )
read_56_disablecopyonread_count: 
savev2_const
identity_115ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_24/DisableCopyOnReadбRead_24/ReadVariableOpбRead_25/DisableCopyOnReadбRead_25/ReadVariableOpбRead_26/DisableCopyOnReadбRead_26/ReadVariableOpбRead_27/DisableCopyOnReadбRead_27/ReadVariableOpбRead_28/DisableCopyOnReadбRead_28/ReadVariableOpбRead_29/DisableCopyOnReadбRead_29/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_30/DisableCopyOnReadбRead_30/ReadVariableOpбRead_31/DisableCopyOnReadбRead_31/ReadVariableOpбRead_32/DisableCopyOnReadбRead_32/ReadVariableOpбRead_33/DisableCopyOnReadбRead_33/ReadVariableOpбRead_34/DisableCopyOnReadбRead_34/ReadVariableOpбRead_35/DisableCopyOnReadбRead_35/ReadVariableOpбRead_36/DisableCopyOnReadбRead_36/ReadVariableOpбRead_37/DisableCopyOnReadбRead_37/ReadVariableOpбRead_38/DisableCopyOnReadбRead_38/ReadVariableOpбRead_39/DisableCopyOnReadбRead_39/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_40/DisableCopyOnReadбRead_40/ReadVariableOpбRead_41/DisableCopyOnReadбRead_41/ReadVariableOpбRead_42/DisableCopyOnReadбRead_42/ReadVariableOpбRead_43/DisableCopyOnReadбRead_43/ReadVariableOpбRead_44/DisableCopyOnReadбRead_44/ReadVariableOpбRead_45/DisableCopyOnReadбRead_45/ReadVariableOpбRead_46/DisableCopyOnReadбRead_46/ReadVariableOpбRead_47/DisableCopyOnReadбRead_47/ReadVariableOpбRead_48/DisableCopyOnReadбRead_48/ReadVariableOpбRead_49/DisableCopyOnReadбRead_49/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_50/DisableCopyOnReadбRead_50/ReadVariableOpбRead_51/DisableCopyOnReadбRead_51/ReadVariableOpбRead_52/DisableCopyOnReadбRead_52/ReadVariableOpбRead_53/DisableCopyOnReadбRead_53/ReadVariableOpбRead_54/DisableCopyOnReadбRead_54/ReadVariableOpбRead_55/DisableCopyOnReadбRead_55/ReadVariableOpбRead_56/DisableCopyOnReadбRead_56/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 б
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_19_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 б
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_19_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 е
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_20_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 б
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_20_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 е
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_21_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 б
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_21_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 е
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_22_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 б
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_22_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:Ё
Read_8/DisableCopyOnReadDisableCopyOnRead1read_8_disablecopyonread_lstm_21_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_8/ReadVariableOpReadVariableOp1read_8_disablecopyonread_lstm_21_lstm_cell_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ј
Read_9/DisableCopyOnReadDisableCopyOnRead;read_9_disablecopyonread_lstm_21_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_9/ReadVariableOpReadVariableOp;read_9_disablecopyonread_lstm_21_lstm_cell_recurrent_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ё
Read_10/DisableCopyOnReadDisableCopyOnRead0read_10_disablecopyonread_lstm_21_lstm_cell_bias"/device:CPU:0*
_output_shapes
 «
Read_10/ReadVariableOpReadVariableOp0read_10_disablecopyonread_lstm_21_lstm_cell_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@Є
Read_11/DisableCopyOnReadDisableCopyOnRead2read_11_disablecopyonread_lstm_22_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_11/ReadVariableOpReadVariableOp2read_11_disablecopyonread_lstm_22_lstm_cell_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@Љ
Read_12/DisableCopyOnReadDisableCopyOnRead<read_12_disablecopyonread_lstm_22_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Й
Read_12/ReadVariableOpReadVariableOp<read_12_disablecopyonread_lstm_22_lstm_cell_recurrent_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ё
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_lstm_22_lstm_cell_bias"/device:CPU:0*
_output_shapes
 «
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_lstm_22_lstm_cell_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@Є
Read_14/DisableCopyOnReadDisableCopyOnRead2read_14_disablecopyonread_lstm_23_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_14/ReadVariableOpReadVariableOp2read_14_disablecopyonread_lstm_23_lstm_cell_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@Љ
Read_15/DisableCopyOnReadDisableCopyOnRead<read_15_disablecopyonread_lstm_23_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Й
Read_15/ReadVariableOpReadVariableOp<read_15_disablecopyonread_lstm_23_lstm_cell_recurrent_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ё
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_lstm_23_lstm_cell_bias"/device:CPU:0*
_output_shapes
 «
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_lstm_23_lstm_cell_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@x
Read_17/DisableCopyOnReadDisableCopyOnRead#read_17_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ю
Read_17/ReadVariableOpReadVariableOp#read_17_disablecopyonread_iteration^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_18/DisableCopyOnReadDisableCopyOnRead'read_18_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 А
Read_18/ReadVariableOpReadVariableOp'read_18_disablecopyonread_learning_rate^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: ј
Read_19/DisableCopyOnReadDisableCopyOnRead9read_19_disablecopyonread_adam_m_lstm_21_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_19/ReadVariableOpReadVariableOp9read_19_disablecopyonread_adam_m_lstm_21_lstm_cell_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:@ј
Read_20/DisableCopyOnReadDisableCopyOnRead9read_20_disablecopyonread_adam_v_lstm_21_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_20/ReadVariableOpReadVariableOp9read_20_disablecopyonread_adam_v_lstm_21_lstm_cell_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_21/DisableCopyOnReadDisableCopyOnReadCread_21_disablecopyonread_adam_m_lstm_21_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_21/ReadVariableOpReadVariableOpCread_21_disablecopyonread_adam_m_lstm_21_lstm_cell_recurrent_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_22/DisableCopyOnReadDisableCopyOnReadCread_22_disablecopyonread_adam_v_lstm_21_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_22/ReadVariableOpReadVariableOpCread_22_disablecopyonread_adam_v_lstm_21_lstm_cell_recurrent_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:@ї
Read_23/DisableCopyOnReadDisableCopyOnRead7read_23_disablecopyonread_adam_m_lstm_21_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_23/ReadVariableOpReadVariableOp7read_23_disablecopyonread_adam_m_lstm_21_lstm_cell_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@ї
Read_24/DisableCopyOnReadDisableCopyOnRead7read_24_disablecopyonread_adam_v_lstm_21_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_24/ReadVariableOpReadVariableOp7read_24_disablecopyonread_adam_v_lstm_21_lstm_cell_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@ј
Read_25/DisableCopyOnReadDisableCopyOnRead9read_25_disablecopyonread_adam_m_lstm_22_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_25/ReadVariableOpReadVariableOp9read_25_disablecopyonread_adam_m_lstm_22_lstm_cell_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:@ј
Read_26/DisableCopyOnReadDisableCopyOnRead9read_26_disablecopyonread_adam_v_lstm_22_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_26/ReadVariableOpReadVariableOp9read_26_disablecopyonread_adam_v_lstm_22_lstm_cell_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_27/DisableCopyOnReadDisableCopyOnReadCread_27_disablecopyonread_adam_m_lstm_22_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_27/ReadVariableOpReadVariableOpCread_27_disablecopyonread_adam_m_lstm_22_lstm_cell_recurrent_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_28/DisableCopyOnReadDisableCopyOnReadCread_28_disablecopyonread_adam_v_lstm_22_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_28/ReadVariableOpReadVariableOpCread_28_disablecopyonread_adam_v_lstm_22_lstm_cell_recurrent_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:@ї
Read_29/DisableCopyOnReadDisableCopyOnRead7read_29_disablecopyonread_adam_m_lstm_22_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_29/ReadVariableOpReadVariableOp7read_29_disablecopyonread_adam_m_lstm_22_lstm_cell_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@ї
Read_30/DisableCopyOnReadDisableCopyOnRead7read_30_disablecopyonread_adam_v_lstm_22_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_30/ReadVariableOpReadVariableOp7read_30_disablecopyonread_adam_v_lstm_22_lstm_cell_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@ј
Read_31/DisableCopyOnReadDisableCopyOnRead9read_31_disablecopyonread_adam_m_lstm_23_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_31/ReadVariableOpReadVariableOp9read_31_disablecopyonread_adam_m_lstm_23_lstm_cell_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:@ј
Read_32/DisableCopyOnReadDisableCopyOnRead9read_32_disablecopyonread_adam_v_lstm_23_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_32/ReadVariableOpReadVariableOp9read_32_disablecopyonread_adam_v_lstm_23_lstm_cell_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_33/DisableCopyOnReadDisableCopyOnReadCread_33_disablecopyonread_adam_m_lstm_23_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_33/ReadVariableOpReadVariableOpCread_33_disablecopyonread_adam_m_lstm_23_lstm_cell_recurrent_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@ў
Read_34/DisableCopyOnReadDisableCopyOnReadCread_34_disablecopyonread_adam_v_lstm_23_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_34/ReadVariableOpReadVariableOpCread_34_disablecopyonread_adam_v_lstm_23_lstm_cell_recurrent_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:@ї
Read_35/DisableCopyOnReadDisableCopyOnRead7read_35_disablecopyonread_adam_m_lstm_23_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_35/ReadVariableOpReadVariableOp7read_35_disablecopyonread_adam_m_lstm_23_lstm_cell_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@ї
Read_36/DisableCopyOnReadDisableCopyOnRead7read_36_disablecopyonread_adam_v_lstm_23_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_36/ReadVariableOpReadVariableOp7read_36_disablecopyonread_adam_v_lstm_23_lstm_cell_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ё
Read_37/DisableCopyOnReadDisableCopyOnRead0read_37_disablecopyonread_adam_m_dense_19_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_37/ReadVariableOpReadVariableOp0read_37_disablecopyonread_adam_m_dense_19_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:Ё
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_v_dense_19_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_v_dense_19_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:Ѓ
Read_39/DisableCopyOnReadDisableCopyOnRead.read_39_disablecopyonread_adam_m_dense_19_bias"/device:CPU:0*
_output_shapes
 г
Read_39/ReadVariableOpReadVariableOp.read_39_disablecopyonread_adam_m_dense_19_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_v_dense_19_bias"/device:CPU:0*
_output_shapes
 г
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_v_dense_19_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:Ё
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_m_dense_20_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:Ё
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_v_dense_20_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:Ѓ
Read_43/DisableCopyOnReadDisableCopyOnRead.read_43_disablecopyonread_adam_m_dense_20_bias"/device:CPU:0*
_output_shapes
 г
Read_43/ReadVariableOpReadVariableOp.read_43_disablecopyonread_adam_m_dense_20_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_v_dense_20_bias"/device:CPU:0*
_output_shapes
 г
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_v_dense_20_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:Ё
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_m_dense_21_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:Ё
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_v_dense_21_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:Ѓ
Read_47/DisableCopyOnReadDisableCopyOnRead.read_47_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 г
Read_47/ReadVariableOpReadVariableOp.read_47_disablecopyonread_adam_m_dense_21_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 г
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_v_dense_21_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:Ё
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_m_dense_22_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:Ё
Read_50/DisableCopyOnReadDisableCopyOnRead0read_50_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_50/ReadVariableOpReadVariableOp0read_50_disablecopyonread_adam_v_dense_22_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:Ѓ
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 г
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_m_dense_22_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_52/DisableCopyOnReadDisableCopyOnRead.read_52_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 г
Read_52/ReadVariableOpReadVariableOp.read_52_disablecopyonread_adam_v_dense_22_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_53/DisableCopyOnReadDisableCopyOnRead!read_53_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Џ
Read_53/ReadVariableOpReadVariableOp!read_53_disablecopyonread_total_1^Read_53/DisableCopyOnRead"/device:CPU:0*
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
: v
Read_54/DisableCopyOnReadDisableCopyOnRead!read_54_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Џ
Read_54/ReadVariableOpReadVariableOp!read_54_disablecopyonread_count_1^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_55/DisableCopyOnReadDisableCopyOnReadread_55_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ў
Read_55/ReadVariableOpReadVariableOpread_55_disablecopyonread_total^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_56/DisableCopyOnReadDisableCopyOnReadread_56_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ў
Read_56/ReadVariableOpReadVariableOpread_56_disablecopyonread_count^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: ╚
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*ы
valueуBС:B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Є
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_114Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_115IdentityIdentity_114:output:0^NoOp*
T0*
_output_shapes
: В
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_115Identity_115:output:0*(
_construction_contextkEagerRuntime*Ѕ
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=:9

_output_shapes
: 

_user_specified_nameConst:%9!

_user_specified_namecount:%8!

_user_specified_nametotal:'7#
!
_user_specified_name	count_1:'6#
!
_user_specified_name	total_1:450
.
_user_specified_nameAdam/v/dense_22/bias:440
.
_user_specified_nameAdam/m/dense_22/bias:632
0
_user_specified_nameAdam/v/dense_22/kernel:622
0
_user_specified_nameAdam/m/dense_22/kernel:410
.
_user_specified_nameAdam/v/dense_21/bias:400
.
_user_specified_nameAdam/m/dense_21/bias:6/2
0
_user_specified_nameAdam/v/dense_21/kernel:6.2
0
_user_specified_nameAdam/m/dense_21/kernel:4-0
.
_user_specified_nameAdam/v/dense_20/bias:4,0
.
_user_specified_nameAdam/m/dense_20/bias:6+2
0
_user_specified_nameAdam/v/dense_20/kernel:6*2
0
_user_specified_nameAdam/m/dense_20/kernel:4)0
.
_user_specified_nameAdam/v/dense_19/bias:4(0
.
_user_specified_nameAdam/m/dense_19/bias:6'2
0
_user_specified_nameAdam/v/dense_19/kernel:6&2
0
_user_specified_nameAdam/m/dense_19/kernel:=%9
7
_user_specified_nameAdam/v/lstm_23/lstm_cell/bias:=$9
7
_user_specified_nameAdam/m/lstm_23/lstm_cell/bias:I#E
C
_user_specified_name+)Adam/v/lstm_23/lstm_cell/recurrent_kernel:I"E
C
_user_specified_name+)Adam/m/lstm_23/lstm_cell/recurrent_kernel:?!;
9
_user_specified_name!Adam/v/lstm_23/lstm_cell/kernel:? ;
9
_user_specified_name!Adam/m/lstm_23/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_22/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_22/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_22/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_22/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_22/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_22/lstm_cell/kernel:=9
7
_user_specified_nameAdam/v/lstm_21/lstm_cell/bias:=9
7
_user_specified_nameAdam/m/lstm_21/lstm_cell/bias:IE
C
_user_specified_name+)Adam/v/lstm_21/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/lstm_21/lstm_cell/recurrent_kernel:?;
9
_user_specified_name!Adam/v/lstm_21/lstm_cell/kernel:?;
9
_user_specified_name!Adam/m/lstm_21/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:62
0
_user_specified_namelstm_23/lstm_cell/bias:B>
<
_user_specified_name$"lstm_23/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_23/lstm_cell/kernel:62
0
_user_specified_namelstm_22/lstm_cell/bias:B>
<
_user_specified_name$"lstm_22/lstm_cell/recurrent_kernel:84
2
_user_specified_namelstm_22/lstm_cell/kernel:62
0
_user_specified_namelstm_21/lstm_cell/bias:B
>
<
_user_specified_name$"lstm_21/lstm_cell/recurrent_kernel:8	4
2
_user_specified_namelstm_21/lstm_cell/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:-)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
џ

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_507706

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
└I
є
C__inference_lstm_21_layer_call_and_return_conditional_losses_509134

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509050*
condR
while_cond_509049*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
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
└I
є
C__inference_lstm_22_layer_call_and_return_conditional_losses_509777

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509693*
condR
while_cond_509692*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
┬	
├
while_cond_509406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509406___redundant_placeholder04
0while_while_cond_509406___redundant_placeholder14
0while_while_cond_509406___redundant_placeholder24
0while_while_cond_509406___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
щ9
»
while_body_510343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
╦

ш
D__inference_dense_19_layer_call_and_return_conditional_losses_510475

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
┬	
├
while_cond_509907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509907___redundant_placeholder04
0while_while_cond_509907___redundant_placeholder14
0while_while_cond_509907___redundant_placeholder24
0while_while_cond_509907___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
л
▓
(__inference_lstm_22_layer_call_fn_509194

inputs
unknown:@
	unknown_0:@
	unknown_1:@
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
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_507435s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name509190:&"
 
_user_specified_name509188:&"
 
_user_specified_name509186:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
└I
є
C__inference_lstm_22_layer_call_and_return_conditional_losses_508025

inputs:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507941*
condR
while_cond_507940*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         *
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
:         *
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
[
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
Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2D
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

 
_user_specified_nameinputs
П8
»
while_body_508764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_510544

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
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
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
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_509549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509549___redundant_placeholder04
0while_while_cond_509549___redundant_placeholder14
0while_while_cond_509549___redundant_placeholder24
0while_while_cond_509549___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
П8
»
while_body_507188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
­
*__inference_lstm_cell_layer_call_fn_510632

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510624:&"
 
_user_specified_name510622:&"
 
_user_specified_name510620:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ж
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_508037

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
■I
ѕ
C__inference_lstm_22_layer_call_and_return_conditional_losses_509348
inputs_0:
(lstm_cell_matmul_readvariableop_resource:@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_509264*
condR
while_cond_509263*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
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
 :                  
"
_user_specified_name
inputs_0
Х
­
*__inference_lstm_cell_layer_call_fn_510747

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
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
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506638o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510739:&"
 
_user_specified_name510737:&"
 
_user_specified_name510735:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀
d
+__inference_dropout_33_layer_call_fn_509139

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
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_507291s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬	
├
while_cond_507514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_507514___redundant_placeholder04
0while_while_cond_507514___redundant_placeholder14
0while_while_cond_507514___redundant_placeholder24
0while_while_cond_507514___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
А
G
+__inference_dropout_37_layer_call_fn_510532

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_508217`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬	
├
while_cond_509692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509692___redundant_placeholder04
0while_while_cond_509692___redundant_placeholder14
0while_while_cond_509692___redundant_placeholder24
0while_while_cond_509692___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
■;
ј
H__inference_sequential_7_layer_call_and_return_conditional_losses_508236
lstm_21_input 
lstm_21_507870:@ 
lstm_21_507872:@
lstm_21_507874:@ 
lstm_22_508026:@ 
lstm_22_508028:@
lstm_22_508030:@ 
lstm_23_508184:@ 
lstm_23_508186:@
lstm_23_508188:@!
dense_19_508197:
dense_19_508199:!
dense_20_508208:
dense_20_508210:!
dense_21_508219:
dense_21_508221:!
dense_22_508230:
dense_22_508232:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЅ
lstm_21/StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputlstm_21_507870lstm_21_507872lstm_21_507874*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_507869Р
dropout_33/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_507881Ъ
lstm_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0lstm_22_508026lstm_22_508028lstm_22_508030*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_508025Р
dropout_34/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_508037Џ
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0lstm_23_508184lstm_23_508186lstm_23_508188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_508183я
dropout_35/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_508195Ї
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0dense_19_508197dense_19_508199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_507631▀
dropout_36/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_508206Ї
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0dense_20_508208dense_20_508210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_507660▀
dropout_37/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_508217Ї
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_21_508219dense_21_508221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_507689▀
dropout_38/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_508228Ї
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_22_508230dense_22_508232*
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
D__inference_dense_22_layer_call_and_return_conditional_losses_507717x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ћ
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         
: : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:&"
 
_user_specified_name508232:&"
 
_user_specified_name508230:&"
 
_user_specified_name508221:&"
 
_user_specified_name508219:&"
 
_user_specified_name508210:&"
 
_user_specified_name508208:&"
 
_user_specified_name508199:&
"
 
_user_specified_name508197:&	"
 
_user_specified_name508188:&"
 
_user_specified_name508186:&"
 
_user_specified_name508184:&"
 
_user_specified_name508030:&"
 
_user_specified_name508028:&"
 
_user_specified_name508026:&"
 
_user_specified_name507874:&"
 
_user_specified_name507872:&"
 
_user_specified_name507870:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_21_input
╝$
╚
while_body_506507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_506531_0:@*
while_lstm_cell_506533_0:@&
while_lstm_cell_506535_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_506531:@(
while_lstm_cell_506533:@$
while_lstm_cell_506535:@ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ц
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_506531_0while_lstm_cell_506533_0while_lstm_cell_506535_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506493┘
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
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_506531while_lstm_cell_506531_0"2
while_lstm_cell_506533while_lstm_cell_506533_0"2
while_lstm_cell_506535while_lstm_cell_506535_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name506535:&	"
 
_user_specified_name506533:&"
 
_user_specified_name506531:_[
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
:         :-)
'
_output_shapes
:         :
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
┬S
л
&sequential_7_lstm_23_while_body_505969F
Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counterL
Hsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations*
&sequential_7_lstm_23_while_placeholder,
(sequential_7_lstm_23_while_placeholder_1,
(sequential_7_lstm_23_while_placeholder_2,
(sequential_7_lstm_23_while_placeholder_3E
Asequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1_0Ђ
}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0W
Esequential_7_lstm_23_while_lstm_cell_matmul_readvariableop_resource_0:@Y
Gsequential_7_lstm_23_while_lstm_cell_matmul_1_readvariableop_resource_0:@T
Fsequential_7_lstm_23_while_lstm_cell_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_23_while_identity)
%sequential_7_lstm_23_while_identity_1)
%sequential_7_lstm_23_while_identity_2)
%sequential_7_lstm_23_while_identity_3)
%sequential_7_lstm_23_while_identity_4)
%sequential_7_lstm_23_while_identity_5C
?sequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1
{sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensorU
Csequential_7_lstm_23_while_lstm_cell_matmul_readvariableop_resource:@W
Esequential_7_lstm_23_while_lstm_cell_matmul_1_readvariableop_resource:@R
Dsequential_7_lstm_23_while_lstm_cell_biasadd_readvariableop_resource:@ѕб;sequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOpб:sequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOpб<sequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_23_while_placeholderUsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0└
:sequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpEsequential_7_lstm_23_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ы
+sequential_7/lstm_23/while/lstm_cell/MatMulMatMulEsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @─
<sequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpGsequential_7_lstm_23_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0┘
-sequential_7/lstm_23/while/lstm_cell/MatMul_1MatMul(sequential_7_lstm_23_while_placeholder_2Dsequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @М
(sequential_7/lstm_23/while/lstm_cell/addAddV25sequential_7/lstm_23/while/lstm_cell/MatMul:product:07sequential_7/lstm_23/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @Й
;sequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpFsequential_7_lstm_23_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0▄
,sequential_7/lstm_23/while/lstm_cell/BiasAddBiasAdd,sequential_7/lstm_23/while/lstm_cell/add:z:0Csequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
4sequential_7/lstm_23/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_7/lstm_23/while/lstm_cell/splitSplit=sequential_7/lstm_23/while/lstm_cell/split/split_dim:output:05sequential_7/lstm_23/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitъ
,sequential_7/lstm_23/while/lstm_cell/SigmoidSigmoid3sequential_7/lstm_23/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_23/while/lstm_cell/Sigmoid_1Sigmoid3sequential_7/lstm_23/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ┐
(sequential_7/lstm_23/while/lstm_cell/mulMul2sequential_7/lstm_23/while/lstm_cell/Sigmoid_1:y:0(sequential_7_lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         ў
)sequential_7/lstm_23/while/lstm_cell/ReluRelu3sequential_7/lstm_23/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ╬
*sequential_7/lstm_23/while/lstm_cell/mul_1Mul0sequential_7/lstm_23/while/lstm_cell/Sigmoid:y:07sequential_7/lstm_23/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ├
*sequential_7/lstm_23/while/lstm_cell/add_1AddV2,sequential_7/lstm_23/while/lstm_cell/mul:z:0.sequential_7/lstm_23/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         а
.sequential_7/lstm_23/while/lstm_cell/Sigmoid_2Sigmoid3sequential_7/lstm_23/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ћ
+sequential_7/lstm_23/while/lstm_cell/Relu_1Relu.sequential_7/lstm_23/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         м
*sequential_7/lstm_23/while/lstm_cell/mul_2Mul2sequential_7/lstm_23/while/lstm_cell/Sigmoid_2:y:09sequential_7/lstm_23/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         Є
Esequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
?sequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_23_while_placeholder_1Nsequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItem/index:output:0.sequential_7/lstm_23/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_23/while/addAddV2&sequential_7_lstm_23_while_placeholder)sequential_7/lstm_23/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_23/while/add_1AddV2Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counter+sequential_7/lstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_23/while/IdentityIdentity$sequential_7/lstm_23/while/add_1:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_23/while/Identity_1IdentityHsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_23/while/Identity_2Identity"sequential_7/lstm_23/while/add:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_23/while/Identity_3IdentityOsequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: х
%sequential_7/lstm_23/while/Identity_4Identity.sequential_7/lstm_23/while/lstm_cell/mul_2:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*'
_output_shapes
:         х
%sequential_7/lstm_23/while/Identity_5Identity.sequential_7/lstm_23/while/lstm_cell/add_1:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*'
_output_shapes
:         э
sequential_7/lstm_23/while/NoOpNoOp<^sequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOp;^sequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOp=^sequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "W
%sequential_7_lstm_23_while_identity_1.sequential_7/lstm_23/while/Identity_1:output:0"W
%sequential_7_lstm_23_while_identity_2.sequential_7/lstm_23/while/Identity_2:output:0"W
%sequential_7_lstm_23_while_identity_3.sequential_7/lstm_23/while/Identity_3:output:0"W
%sequential_7_lstm_23_while_identity_4.sequential_7/lstm_23/while/Identity_4:output:0"W
%sequential_7_lstm_23_while_identity_5.sequential_7/lstm_23/while/Identity_5:output:0"S
#sequential_7_lstm_23_while_identity,sequential_7/lstm_23/while/Identity:output:0"ј
Dsequential_7_lstm_23_while_lstm_cell_biasadd_readvariableop_resourceFsequential_7_lstm_23_while_lstm_cell_biasadd_readvariableop_resource_0"љ
Esequential_7_lstm_23_while_lstm_cell_matmul_1_readvariableop_resourceGsequential_7_lstm_23_while_lstm_cell_matmul_1_readvariableop_resource_0"ї
Csequential_7_lstm_23_while_lstm_cell_matmul_readvariableop_resourceEsequential_7_lstm_23_while_lstm_cell_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1Asequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1_0"Ч
{sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2z
;sequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOp;sequential_7/lstm_23/while/lstm_cell/BiasAdd/ReadVariableOp2x
:sequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOp:sequential_7/lstm_23/while/lstm_cell/MatMul/ReadVariableOp2|
<sequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOp<sequential_7/lstm_23/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:tp

_output_shapes
: 
V
_user_specified_name><sequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor:\X

_output_shapes
: 
>
_user_specified_name&$sequential_7/lstm_23/strided_slice_1:-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :ea

_output_shapes
: 
G
_user_specified_name/-sequential_7/lstm_23/while/maximum_iterations:_ [

_output_shapes
: 
A
_user_specified_name)'sequential_7/lstm_23/while/loop_counter
а9
Ш
C__inference_lstm_23_layer_call_and_return_conditional_losses_507071

inputs"
lstm_cell_506987:@"
lstm_cell_506989:@
lstm_cell_506991:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
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
valueB"       Я
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
:         *
shrink_axis_maskТ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506987lstm_cell_506989lstm_cell_506991*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506986n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506987lstm_cell_506989lstm_cell_506991*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507001*
condR
while_cond_507000*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506991:&"
 
_user_specified_name506989:&"
 
_user_specified_name506987:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Е
ђ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510681

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
"
_user_specified_name
states_1:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь
ќ
)__inference_dense_21_layer_call_fn_510558

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_507689o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name510554:&"
 
_user_specified_name510552:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_510502

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х8
Ш
C__inference_lstm_21_layer_call_and_return_conditional_losses_506230

inputs"
lstm_cell_506148:@"
lstm_cell_506150:@
lstm_cell_506152:@
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
value	B :s
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
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:         c
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
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_506148lstm_cell_506150lstm_cell_506152*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506147n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_506148lstm_cell_506150lstm_cell_506152*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506161*
condR
while_cond_506160*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
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
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name506152:&"
 
_user_specified_name506150:&"
 
_user_specified_name506148:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
щ9
»
while_body_507515
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         r
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
while_cond_509263
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_509263___redundant_placeholder04
0while_while_cond_509263___redundant_placeholder14
0while_while_cond_509263___redundant_placeholder24
0while_while_cond_509263___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
л
▓
(__inference_lstm_21_layer_call_fn_508551

inputs
unknown:@
	unknown_0:@
	unknown_1:@
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
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_507272s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name508547:&"
 
_user_specified_name508545:&"
 
_user_specified_name508543:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┬	
├
while_cond_506853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_506853___redundant_placeholder04
0while_while_cond_506853___redundant_placeholder14
0while_while_cond_506853___redundant_placeholder24
0while_while_cond_506853___redundant_placeholder3
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
@: : : : :         :         : :::::
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
:         :-)
'
_output_shapes
:         :
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
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506986

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
А
G
+__inference_dropout_35_layer_call_fn_510438

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
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_508195`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
А
■
E__inference_lstm_cell_layer_call_and_return_conditional_losses_506493

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
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
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
П8
»
while_body_508907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
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
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :
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
_user_specified_namewhile/loop_counter"ДL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultД
K
lstm_21_input:
serving_default_lstm_21_input:0         
<
dense_220
StatefulPartitionedCall:0         tensorflow/serving/predict:ћч
╚
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
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╝
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator"
_tf_keras_layer
┌
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator
.cell
/
state_spec"
_tf_keras_rnn_layer
╝
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
┌
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator
>cell
?
state_spec"
_tf_keras_rnn_layer
╝
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator"
_tf_keras_layer
╗
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias"
_tf_keras_layer
╝
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator"
_tf_keras_layer
╗
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
╝
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d_random_generator"
_tf_keras_layer
╗
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
╝
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator"
_tf_keras_layer
╗
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
Б
|0
}1
~2
3
ђ4
Ђ5
ѓ6
Ѓ7
ё8
M9
N10
\11
]12
k13
l14
z15
{16"
trackable_list_wrapper
Б
|0
}1
~2
3
ђ4
Ђ5
ѓ6
Ѓ7
ё8
M9
N10
\11
]12
k13
l14
z15
{16"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Л
іtrace_0
Іtrace_12ќ
-__inference_sequential_7_layer_call_fn_508275
-__inference_sequential_7_layer_call_fn_508314х
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
 zіtrace_0zІtrace_1
Є
їtrace_0
Їtrace_12╠
H__inference_sequential_7_layer_call_and_return_conditional_losses_507724
H__inference_sequential_7_layer_call_and_return_conditional_losses_508236х
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
 zїtrace_0zЇtrace_1
мB¤
!__inference__wrapped_model_506085lstm_21_input"ў
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
ј
_variables
Ј_iterations
љ_learning_rate
Љ_index_dict
њ
_momentums
Њ_velocities
ћ_update_step_xla"
experimentalOptimizer
-
Ћserving_default"
signature_map
5
|0
}1
~2"
trackable_list_wrapper
5
|0
}1
~2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
ќstates
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
У
юtrace_0
Юtrace_1
ъtrace_2
Ъtrace_32ш
(__inference_lstm_21_layer_call_fn_508529
(__inference_lstm_21_layer_call_fn_508540
(__inference_lstm_21_layer_call_fn_508551
(__inference_lstm_21_layer_call_fn_508562╩
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
 zюtrace_0zЮtrace_1zъtrace_2zЪtrace_3
н
аtrace_0
Аtrace_1
бtrace_2
Бtrace_32р
C__inference_lstm_21_layer_call_and_return_conditional_losses_508705
C__inference_lstm_21_layer_call_and_return_conditional_losses_508848
C__inference_lstm_21_layer_call_and_return_conditional_losses_508991
C__inference_lstm_21_layer_call_and_return_conditional_losses_509134╩
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
 zаtrace_0zАtrace_1zбtrace_2zБtrace_3
"
_generic_user_object
ђ
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
ф_random_generator
Ф
state_size

|kernel
}recurrent_kernel
~bias"
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
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
┴
▒trace_0
▓trace_12є
+__inference_dropout_33_layer_call_fn_509139
+__inference_dropout_33_layer_call_fn_509144Е
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
 z▒trace_0z▓trace_1
э
│trace_0
┤trace_12╝
F__inference_dropout_33_layer_call_and_return_conditional_losses_509156
F__inference_dropout_33_layer_call_and_return_conditional_losses_509161Е
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
 z│trace_0z┤trace_1
"
_generic_user_object
7
0
ђ1
Ђ2"
trackable_list_wrapper
7
0
ђ1
Ђ2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
хstates
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
У
╗trace_0
╝trace_1
йtrace_2
Йtrace_32ш
(__inference_lstm_22_layer_call_fn_509172
(__inference_lstm_22_layer_call_fn_509183
(__inference_lstm_22_layer_call_fn_509194
(__inference_lstm_22_layer_call_fn_509205╩
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
 z╗trace_0z╝trace_1zйtrace_2zЙtrace_3
н
┐trace_0
└trace_1
┴trace_2
┬trace_32р
C__inference_lstm_22_layer_call_and_return_conditional_losses_509348
C__inference_lstm_22_layer_call_and_return_conditional_losses_509491
C__inference_lstm_22_layer_call_and_return_conditional_losses_509634
C__inference_lstm_22_layer_call_and_return_conditional_losses_509777╩
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
 z┐trace_0z└trace_1z┴trace_2z┬trace_3
"
_generic_user_object
ѓ
├	variables
─trainable_variables
┼regularization_losses
к	keras_api
К__call__
+╚&call_and_return_all_conditional_losses
╔_random_generator
╩
state_size

kernel
ђrecurrent_kernel
	Ђbias"
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
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
┴
лtrace_0
Лtrace_12є
+__inference_dropout_34_layer_call_fn_509782
+__inference_dropout_34_layer_call_fn_509787Е
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
 zлtrace_0zЛtrace_1
э
мtrace_0
Мtrace_12╝
F__inference_dropout_34_layer_call_and_return_conditional_losses_509799
F__inference_dropout_34_layer_call_and_return_conditional_losses_509804Е
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
 zмtrace_0zМtrace_1
"
_generic_user_object
8
ѓ0
Ѓ1
ё2"
trackable_list_wrapper
8
ѓ0
Ѓ1
ё2"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
нstates
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
У
┌trace_0
█trace_1
▄trace_2
Пtrace_32ш
(__inference_lstm_23_layer_call_fn_509815
(__inference_lstm_23_layer_call_fn_509826
(__inference_lstm_23_layer_call_fn_509837
(__inference_lstm_23_layer_call_fn_509848╩
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
 z┌trace_0z█trace_1z▄trace_2zПtrace_3
н
яtrace_0
▀trace_1
Яtrace_2
рtrace_32р
C__inference_lstm_23_layer_call_and_return_conditional_losses_509993
C__inference_lstm_23_layer_call_and_return_conditional_losses_510138
C__inference_lstm_23_layer_call_and_return_conditional_losses_510283
C__inference_lstm_23_layer_call_and_return_conditional_losses_510428╩
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
 zяtrace_0z▀trace_1zЯtrace_2zрtrace_3
"
_generic_user_object
Ѓ
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses
У_random_generator
ж
state_size
ѓkernel
Ѓrecurrent_kernel
	ёbias"
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
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
┴
№trace_0
­trace_12є
+__inference_dropout_35_layer_call_fn_510433
+__inference_dropout_35_layer_call_fn_510438Е
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
 z№trace_0z­trace_1
э
ыtrace_0
Ыtrace_12╝
F__inference_dropout_35_layer_call_and_return_conditional_losses_510450
F__inference_dropout_35_layer_call_and_return_conditional_losses_510455Е
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
 zыtrace_0zЫtrace_1
"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
т
Эtrace_02к
)__inference_dense_19_layer_call_fn_510464ў
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
 zЭtrace_0
ђ
щtrace_02р
D__inference_dense_19_layer_call_and_return_conditional_losses_510475ў
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
 zщtrace_0
!:2dense_19/kernel
:2dense_19/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Щnon_trainable_variables
чlayers
Чmetrics
 §layer_regularization_losses
■layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
┴
 trace_0
ђtrace_12є
+__inference_dropout_36_layer_call_fn_510480
+__inference_dropout_36_layer_call_fn_510485Е
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
 z trace_0zђtrace_1
э
Ђtrace_0
ѓtrace_12╝
F__inference_dropout_36_layer_call_and_return_conditional_losses_510497
F__inference_dropout_36_layer_call_and_return_conditional_losses_510502Е
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
"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
т
ѕtrace_02к
)__inference_dense_20_layer_call_fn_510511ў
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
 zѕtrace_0
ђ
Ѕtrace_02р
D__inference_dense_20_layer_call_and_return_conditional_losses_510522ў
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
 zЅtrace_0
!:2dense_20/kernel
:2dense_20/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
┴
Јtrace_0
љtrace_12є
+__inference_dropout_37_layer_call_fn_510527
+__inference_dropout_37_layer_call_fn_510532Е
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
 zЈtrace_0zљtrace_1
э
Љtrace_0
њtrace_12╝
F__inference_dropout_37_layer_call_and_return_conditional_losses_510544
F__inference_dropout_37_layer_call_and_return_conditional_losses_510549Е
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
"
_generic_user_object
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
т
ўtrace_02к
)__inference_dense_21_layer_call_fn_510558ў
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
 zўtrace_0
ђ
Ўtrace_02р
D__inference_dense_21_layer_call_and_return_conditional_losses_510569ў
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
 zЎtrace_0
!:2dense_21/kernel
:2dense_21/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
┴
Ъtrace_0
аtrace_12є
+__inference_dropout_38_layer_call_fn_510574
+__inference_dropout_38_layer_call_fn_510579Е
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
 zЪtrace_0zаtrace_1
э
Аtrace_0
бtrace_12╝
F__inference_dropout_38_layer_call_and_return_conditional_losses_510591
F__inference_dropout_38_layer_call_and_return_conditional_losses_510596Е
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
 zАtrace_0zбtrace_1
"
_generic_user_object
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
т
еtrace_02к
)__inference_dense_22_layer_call_fn_510605ў
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
 zеtrace_0
ђ
Еtrace_02р
D__inference_dense_22_layer_call_and_return_conditional_losses_510615ў
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
 zЕtrace_0
!:2dense_22/kernel
:2dense_22/bias
*:(@2lstm_21/lstm_cell/kernel
4:2@2"lstm_21/lstm_cell/recurrent_kernel
$:"@2lstm_21/lstm_cell/bias
*:(@2lstm_22/lstm_cell/kernel
4:2@2"lstm_22/lstm_cell/recurrent_kernel
$:"@2lstm_22/lstm_cell/bias
*:(@2lstm_23/lstm_cell/kernel
4:2@2"lstm_23/lstm_cell/recurrent_kernel
$:"@2lstm_23/lstm_cell/bias
 "
trackable_list_wrapper
~
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
10
11
12"
trackable_list_wrapper
0
ф0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЫB№
-__inference_sequential_7_layer_call_fn_508275lstm_21_input"г
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
ЫB№
-__inference_sequential_7_layer_call_fn_508314lstm_21_input"г
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
ЇBі
H__inference_sequential_7_layer_call_and_return_conditional_losses_507724lstm_21_input"г
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
ЇBі
H__inference_sequential_7_layer_call_and_return_conditional_losses_508236lstm_21_input"г
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
Л
Ј0
г1
Г2
«3
»4
░5
▒6
▓7
│8
┤9
х10
Х11
и12
И13
╣14
║15
╗16
╝17
й18
Й19
┐20
└21
┴22
┬23
├24
─25
┼26
к27
К28
╚29
╔30
╩31
╦32
╠33
═34"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
»
г0
«1
░2
▓3
┤4
Х5
И6
║7
╝8
Й9
└10
┬11
─12
к13
╚14
╩15
╠16"
trackable_list_wrapper
»
Г0
»1
▒2
│3
х4
и5
╣6
╗7
й8
┐9
┴10
├11
┼12
К13
╔14
╦15
═16"
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
$__inference_signature_wrapper_508518lstm_21_input"Ъ
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
jlstm_21_input
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_21_layer_call_fn_508529inputs_0"й
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
(__inference_lstm_21_layer_call_fn_508540inputs_0"й
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
(__inference_lstm_21_layer_call_fn_508551inputs"й
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
(__inference_lstm_21_layer_call_fn_508562inputs"й
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
C__inference_lstm_21_layer_call_and_return_conditional_losses_508705inputs_0"й
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
C__inference_lstm_21_layer_call_and_return_conditional_losses_508848inputs_0"й
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
C__inference_lstm_21_layer_call_and_return_conditional_losses_508991inputs"й
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
C__inference_lstm_21_layer_call_and_return_conditional_losses_509134inputs"й
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
|0
}1
~2"
trackable_list_wrapper
5
|0
}1
~2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
╔
Мtrace_0
нtrace_12ј
*__inference_lstm_cell_layer_call_fn_510632
*__inference_lstm_cell_layer_call_fn_510649│
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
 zМtrace_0zнtrace_1
 
Нtrace_0
оtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510681
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510713│
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
+__inference_dropout_33_layer_call_fn_509139inputs"ц
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
+__inference_dropout_33_layer_call_fn_509144inputs"ц
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
F__inference_dropout_33_layer_call_and_return_conditional_losses_509156inputs"ц
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
F__inference_dropout_33_layer_call_and_return_conditional_losses_509161inputs"ц
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
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_22_layer_call_fn_509172inputs_0"й
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
(__inference_lstm_22_layer_call_fn_509183inputs_0"й
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
(__inference_lstm_22_layer_call_fn_509194inputs"й
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
(__inference_lstm_22_layer_call_fn_509205inputs"й
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_509348inputs_0"й
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_509491inputs_0"й
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_509634inputs"й
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_509777inputs"й
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
7
0
ђ1
Ђ2"
trackable_list_wrapper
7
0
ђ1
Ђ2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
├	variables
─trainable_variables
┼regularization_losses
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
╔
▄trace_0
Пtrace_12ј
*__inference_lstm_cell_layer_call_fn_510730
*__inference_lstm_cell_layer_call_fn_510747│
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
 z▄trace_0zПtrace_1
 
яtrace_0
▀trace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510779
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510811│
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
+__inference_dropout_34_layer_call_fn_509782inputs"ц
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
+__inference_dropout_34_layer_call_fn_509787inputs"ц
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_509799inputs"ц
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_509804inputs"ц
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
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
(__inference_lstm_23_layer_call_fn_509815inputs_0"й
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
(__inference_lstm_23_layer_call_fn_509826inputs_0"й
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
(__inference_lstm_23_layer_call_fn_509837inputs"й
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
(__inference_lstm_23_layer_call_fn_509848inputs"й
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_509993inputs_0"й
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_510138inputs_0"й
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_510283inputs"й
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_510428inputs"й
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
8
ѓ0
Ѓ1
ё2"
trackable_list_wrapper
8
ѓ0
Ѓ1
ё2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
╔
тtrace_0
Тtrace_12ј
*__inference_lstm_cell_layer_call_fn_510828
*__inference_lstm_cell_layer_call_fn_510845│
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
 zтtrace_0zТtrace_1
 
уtrace_0
Уtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510877
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510909│
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
 zуtrace_0zУtrace_1
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
+__inference_dropout_35_layer_call_fn_510433inputs"ц
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
+__inference_dropout_35_layer_call_fn_510438inputs"ц
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_510450inputs"ц
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_510455inputs"ц
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
)__inference_dense_19_layer_call_fn_510464inputs"ў
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
D__inference_dense_19_layer_call_and_return_conditional_losses_510475inputs"ў
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
+__inference_dropout_36_layer_call_fn_510480inputs"ц
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
+__inference_dropout_36_layer_call_fn_510485inputs"ц
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
F__inference_dropout_36_layer_call_and_return_conditional_losses_510497inputs"ц
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
F__inference_dropout_36_layer_call_and_return_conditional_losses_510502inputs"ц
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
)__inference_dense_20_layer_call_fn_510511inputs"ў
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
D__inference_dense_20_layer_call_and_return_conditional_losses_510522inputs"ў
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
+__inference_dropout_37_layer_call_fn_510527inputs"ц
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
+__inference_dropout_37_layer_call_fn_510532inputs"ц
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_510544inputs"ц
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
F__inference_dropout_37_layer_call_and_return_conditional_losses_510549inputs"ц
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
)__inference_dense_21_layer_call_fn_510558inputs"ў
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
D__inference_dense_21_layer_call_and_return_conditional_losses_510569inputs"ў
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
+__inference_dropout_38_layer_call_fn_510574inputs"ц
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
+__inference_dropout_38_layer_call_fn_510579inputs"ц
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
F__inference_dropout_38_layer_call_and_return_conditional_losses_510591inputs"ц
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
F__inference_dropout_38_layer_call_and_return_conditional_losses_510596inputs"ц
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
)__inference_dense_22_layer_call_fn_510605inputs"ў
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
D__inference_dense_22_layer_call_and_return_conditional_losses_510615inputs"ў
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
ж	variables
Ж	keras_api

вtotal

Вcount"
_tf_keras_metric
c
ь	variables
Ь	keras_api

№total

­count
ы
_fn_kwargs"
_tf_keras_metric
/:-@2Adam/m/lstm_21/lstm_cell/kernel
/:-@2Adam/v/lstm_21/lstm_cell/kernel
9:7@2)Adam/m/lstm_21/lstm_cell/recurrent_kernel
9:7@2)Adam/v/lstm_21/lstm_cell/recurrent_kernel
):'@2Adam/m/lstm_21/lstm_cell/bias
):'@2Adam/v/lstm_21/lstm_cell/bias
/:-@2Adam/m/lstm_22/lstm_cell/kernel
/:-@2Adam/v/lstm_22/lstm_cell/kernel
9:7@2)Adam/m/lstm_22/lstm_cell/recurrent_kernel
9:7@2)Adam/v/lstm_22/lstm_cell/recurrent_kernel
):'@2Adam/m/lstm_22/lstm_cell/bias
):'@2Adam/v/lstm_22/lstm_cell/bias
/:-@2Adam/m/lstm_23/lstm_cell/kernel
/:-@2Adam/v/lstm_23/lstm_cell/kernel
9:7@2)Adam/m/lstm_23/lstm_cell/recurrent_kernel
9:7@2)Adam/v/lstm_23/lstm_cell/recurrent_kernel
):'@2Adam/m/lstm_23/lstm_cell/bias
):'@2Adam/v/lstm_23/lstm_cell/bias
&:$2Adam/m/dense_19/kernel
&:$2Adam/v/dense_19/kernel
 :2Adam/m/dense_19/bias
 :2Adam/v/dense_19/bias
&:$2Adam/m/dense_20/kernel
&:$2Adam/v/dense_20/kernel
 :2Adam/m/dense_20/bias
 :2Adam/v/dense_20/bias
&:$2Adam/m/dense_21/kernel
&:$2Adam/v/dense_21/kernel
 :2Adam/m/dense_21/bias
 :2Adam/v/dense_21/bias
&:$2Adam/m/dense_22/kernel
&:$2Adam/v/dense_22/kernel
 :2Adam/m/dense_22/bias
 :2Adam/v/dense_22/bias
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
*__inference_lstm_cell_layer_call_fn_510632inputsstates_0states_1"«
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
*__inference_lstm_cell_layer_call_fn_510649inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510681inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510713inputsstates_0states_1"«
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
*__inference_lstm_cell_layer_call_fn_510730inputsstates_0states_1"«
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
*__inference_lstm_cell_layer_call_fn_510747inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510779inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510811inputsstates_0states_1"«
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
*__inference_lstm_cell_layer_call_fn_510828inputsstates_0states_1"«
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
*__inference_lstm_cell_layer_call_fn_510845inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510877inputsstates_0states_1"«
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510909inputsstates_0states_1"«
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
в0
В1"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
:  (2total
:  (2count
0
№0
­1"
trackable_list_wrapper
.
ь	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper»
!__inference__wrapped_model_506085Ѕ|}~ђЂѓЃёMN\]klz{:б7
0б-
+і(
lstm_21_input         

ф "3ф0
.
dense_22"і
dense_22         Ф
D__inference_dense_19_layer_call_and_return_conditional_losses_510475cMN/б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_19_layer_call_fn_510464XMN/б,
%б"
 і
inputs         
ф "!і
unknown         Ф
D__inference_dense_20_layer_call_and_return_conditional_losses_510522c\]/б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_20_layer_call_fn_510511X\]/б,
%б"
 і
inputs         
ф "!і
unknown         Ф
D__inference_dense_21_layer_call_and_return_conditional_losses_510569ckl/б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_21_layer_call_fn_510558Xkl/б,
%б"
 і
inputs         
ф "!і
unknown         Ф
D__inference_dense_22_layer_call_and_return_conditional_losses_510615cz{/б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_22_layer_call_fn_510605Xz{/б,
%б"
 і
inputs         
ф "!і
unknown         х
F__inference_dropout_33_layer_call_and_return_conditional_losses_509156k7б4
-б*
$і!
inputs         

p
ф "0б-
&і#
tensor_0         

џ х
F__inference_dropout_33_layer_call_and_return_conditional_losses_509161k7б4
-б*
$і!
inputs         

p 
ф "0б-
&і#
tensor_0         

џ Ј
+__inference_dropout_33_layer_call_fn_509139`7б4
-б*
$і!
inputs         

p
ф "%і"
unknown         
Ј
+__inference_dropout_33_layer_call_fn_509144`7б4
-б*
$і!
inputs         

p 
ф "%і"
unknown         
х
F__inference_dropout_34_layer_call_and_return_conditional_losses_509799k7б4
-б*
$і!
inputs         

p
ф "0б-
&і#
tensor_0         

џ х
F__inference_dropout_34_layer_call_and_return_conditional_losses_509804k7б4
-б*
$і!
inputs         

p 
ф "0б-
&і#
tensor_0         

џ Ј
+__inference_dropout_34_layer_call_fn_509782`7б4
-б*
$і!
inputs         

p
ф "%і"
unknown         
Ј
+__inference_dropout_34_layer_call_fn_509787`7б4
-б*
$і!
inputs         

p 
ф "%і"
unknown         
Г
F__inference_dropout_35_layer_call_and_return_conditional_losses_510450c3б0
)б&
 і
inputs         
p
ф ",б)
"і
tensor_0         
џ Г
F__inference_dropout_35_layer_call_and_return_conditional_losses_510455c3б0
)б&
 і
inputs         
p 
ф ",б)
"і
tensor_0         
џ Є
+__inference_dropout_35_layer_call_fn_510433X3б0
)б&
 і
inputs         
p
ф "!і
unknown         Є
+__inference_dropout_35_layer_call_fn_510438X3б0
)б&
 і
inputs         
p 
ф "!і
unknown         Г
F__inference_dropout_36_layer_call_and_return_conditional_losses_510497c3б0
)б&
 і
inputs         
p
ф ",б)
"і
tensor_0         
џ Г
F__inference_dropout_36_layer_call_and_return_conditional_losses_510502c3б0
)б&
 і
inputs         
p 
ф ",б)
"і
tensor_0         
џ Є
+__inference_dropout_36_layer_call_fn_510480X3б0
)б&
 і
inputs         
p
ф "!і
unknown         Є
+__inference_dropout_36_layer_call_fn_510485X3б0
)б&
 і
inputs         
p 
ф "!і
unknown         Г
F__inference_dropout_37_layer_call_and_return_conditional_losses_510544c3б0
)б&
 і
inputs         
p
ф ",б)
"і
tensor_0         
џ Г
F__inference_dropout_37_layer_call_and_return_conditional_losses_510549c3б0
)б&
 і
inputs         
p 
ф ",б)
"і
tensor_0         
џ Є
+__inference_dropout_37_layer_call_fn_510527X3б0
)б&
 і
inputs         
p
ф "!і
unknown         Є
+__inference_dropout_37_layer_call_fn_510532X3б0
)б&
 і
inputs         
p 
ф "!і
unknown         Г
F__inference_dropout_38_layer_call_and_return_conditional_losses_510591c3б0
)б&
 і
inputs         
p
ф ",б)
"і
tensor_0         
џ Г
F__inference_dropout_38_layer_call_and_return_conditional_losses_510596c3б0
)б&
 і
inputs         
p 
ф ",б)
"і
tensor_0         
џ Є
+__inference_dropout_38_layer_call_fn_510574X3б0
)б&
 і
inputs         
p
ф "!і
unknown         Є
+__inference_dropout_38_layer_call_fn_510579X3б0
)б&
 і
inputs         
p 
ф "!і
unknown         ┘
C__inference_lstm_21_layer_call_and_return_conditional_losses_508705Љ|}~OбL
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
tensor_0                  
џ ┘
C__inference_lstm_21_layer_call_and_return_conditional_losses_508848Љ|}~OбL
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
tensor_0                  
џ ┐
C__inference_lstm_21_layer_call_and_return_conditional_losses_508991x|}~?б<
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

џ ┐
C__inference_lstm_21_layer_call_and_return_conditional_losses_509134x|}~?б<
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

џ │
(__inference_lstm_21_layer_call_fn_508529є|}~OбL
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
unknown                  │
(__inference_lstm_21_layer_call_fn_508540є|}~OбL
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
unknown                  Ў
(__inference_lstm_21_layer_call_fn_508551m|}~?б<
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
Ў
(__inference_lstm_21_layer_call_fn_508562m|}~?б<
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
█
C__inference_lstm_22_layer_call_and_return_conditional_losses_509348ЊђЂOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "9б6
/і,
tensor_0                  
џ █
C__inference_lstm_22_layer_call_and_return_conditional_losses_509491ЊђЂOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "9б6
/і,
tensor_0                  
џ ┴
C__inference_lstm_22_layer_call_and_return_conditional_losses_509634zђЂ?б<
5б2
$і!
inputs         


 
p

 
ф "0б-
&і#
tensor_0         

џ ┴
C__inference_lstm_22_layer_call_and_return_conditional_losses_509777zђЂ?б<
5б2
$і!
inputs         


 
p 

 
ф "0б-
&і#
tensor_0         

џ х
(__inference_lstm_22_layer_call_fn_509172ѕђЂOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ".і+
unknown                  х
(__inference_lstm_22_layer_call_fn_509183ѕђЂOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ".і+
unknown                  Џ
(__inference_lstm_22_layer_call_fn_509194oђЂ?б<
5б2
$і!
inputs         


 
p

 
ф "%і"
unknown         
Џ
(__inference_lstm_22_layer_call_fn_509205oђЂ?б<
5б2
$і!
inputs         


 
p 

 
ф "%і"
unknown         
¤
C__inference_lstm_23_layer_call_and_return_conditional_losses_509993ЄѓЃёOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ",б)
"і
tensor_0         
џ ¤
C__inference_lstm_23_layer_call_and_return_conditional_losses_510138ЄѓЃёOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ",б)
"і
tensor_0         
џ Й
C__inference_lstm_23_layer_call_and_return_conditional_losses_510283wѓЃё?б<
5б2
$і!
inputs         


 
p

 
ф ",б)
"і
tensor_0         
џ Й
C__inference_lstm_23_layer_call_and_return_conditional_losses_510428wѓЃё?б<
5б2
$і!
inputs         


 
p 

 
ф ",б)
"і
tensor_0         
џ е
(__inference_lstm_23_layer_call_fn_509815|ѓЃёOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "!і
unknown         е
(__inference_lstm_23_layer_call_fn_509826|ѓЃёOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "!і
unknown         ў
(__inference_lstm_23_layer_call_fn_509837lѓЃё?б<
5б2
$і!
inputs         


 
p

 
ф "!і
unknown         ў
(__inference_lstm_23_layer_call_fn_509848lѓЃё?б<
5б2
$і!
inputs         


 
p 

 
ф "!і
unknown         я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510681ћ|}~ђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510713ћ|}~ђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ Я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510779ќђЂђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ Я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510811ќђЂђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ р
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510877ЌѓЃёђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ р
E__inference_lstm_cell_layer_call_and_return_conditional_losses_510909ЌѓЃёђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ ▒
*__inference_lstm_cell_layer_call_fn_510632ѓ|}~ђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         ▒
*__inference_lstm_cell_layer_call_fn_510649ѓ|}~ђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         │
*__inference_lstm_cell_layer_call_fn_510730ёђЂђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         │
*__inference_lstm_cell_layer_call_fn_510747ёђЂђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         ┤
*__inference_lstm_cell_layer_call_fn_510828ЁѓЃёђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         ┤
*__inference_lstm_cell_layer_call_fn_510845ЁѓЃёђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         О
H__inference_sequential_7_layer_call_and_return_conditional_losses_507724і|}~ђЂѓЃёMN\]klz{Bб?
8б5
+і(
lstm_21_input         

p

 
ф ",б)
"і
tensor_0         
џ О
H__inference_sequential_7_layer_call_and_return_conditional_losses_508236і|}~ђЂѓЃёMN\]klz{Bб?
8б5
+і(
lstm_21_input         

p 

 
ф ",б)
"і
tensor_0         
џ ░
-__inference_sequential_7_layer_call_fn_508275|}~ђЂѓЃёMN\]klz{Bб?
8б5
+і(
lstm_21_input         

p

 
ф "!і
unknown         ░
-__inference_sequential_7_layer_call_fn_508314|}~ђЂѓЃёMN\]klz{Bб?
8б5
+і(
lstm_21_input         

p 

 
ф "!і
unknown         ├
$__inference_signature_wrapper_508518џ|}~ђЂѓЃёMN\]klz{KбH
б 
Aф>
<
lstm_21_input+і(
lstm_21_input         
"3ф0
.
dense_22"і
dense_22         