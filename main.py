from sklearn.cluster import KMeans
import numpy as np


def compute_design_matrix(X, centers, spreads):
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads)*(X - centers), axis = 2)/(-2.0)).T
    return np.insert(basis_func_outputs, 0, 1, axis=1)

def compute_closed_form_sol(L2_lambda, design_matrix, output_data):
    return (np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),np.matmul(design_matrix.T, output_data)).flatten())

def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):

    N  = np.shape(output_data)[0]
    print(N)
    weights = np.zeros([1,11])
    print(design_matrix)
    output_data=np.array(output_data)
    print(output_data)

    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            #print(Phi.shape)
            #print(t.shape)
            #print(weights.shape)
            E_D = np.matmul((np.matmul(Phi, weights.T) - t).T,Phi)
            E = (E_D + L2_lambda * weights)/ minibatch_size
            weights = weights - (learning_rate * E)
        print(np.linalg.norm(E))
    return weights.flatten()

#Read Dataset

letor_input_data = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')

letor_output_data = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

syn_input_data = np.genfromtxt('Syn_X.csv', delimiter = ',')

syn_output_data = np.genfromtxt('Syn_t.csv', delimiter = ',').reshape([-1, 1])


#Partitioning

num_rows_letor = np.shape(letor_input_data)[0]
num_columns_letor = np.shape(letor_input_data)[1]
letor_training_input = []
letor_training_output = []
letor_validation_input = []
letor_validation_output = []
letor_testing_input = []
letor_testing_output = []

#Training Set
for i in range(0, int(0.8*num_rows_letor)):
    letor_training_input.append(letor_input_data[i])
    letor_training_output.append(letor_output_data[i])

#Validation Set
for i in range(int(0.8*num_rows_letor), int(0.9*num_rows_letor)):
    letor_validation_input.append(letor_input_data[i])
    letor_validation_output.append(letor_output_data[i])

#Testing Set
for i in range(int(0.9*num_rows_letor), num_rows_letor):
    letor_testing_input.append(letor_input_data[i])
    letor_testing_output.append(letor_output_data[i])

# Hyper parameters and training
X = np.array(letor_training_input)
M = 10
kmeans = KMeans(n_clusters = M, random_state = 0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

clusters_in_label = {i: X[np.where(labels == i)] for i in range(kmeans.n_clusters)}
spreads = list((np.cov(clusters_in_label[i].T)) for i in range(kmeans.n_clusters))
centers = centers[:,np.newaxis, :]

#Design Matrix
phi = compute_design_matrix(X, centers, spreads)
print(phi)
print(np.shape(phi))

#Closed Form Solution
w = compute_closed_form_sol(0.1, np.array(phi), np.array(letor_training_output, dtype=np.float))
print(w)
closed_form = np.matmul(np.transpose(w), np.transpose(phi))
print(closed_form)
print(np.shape(closed_form))

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxVALIDATIONxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Tuning hyper parameters using validation set
X = np.array(letor_validation_input)
M = 10
kmeans = KMeans(n_clusters = M, random_state = 0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
kmeans_list = {j: X[np.where(labels == j)] for j in range(kmeans.n_clusters)}
spreads = list((np.cov(kmeans_list[j].T)) for j in range(kmeans.n_clusters))
centers = centers[:, np.newaxis, :]

#Design Matrix for Validation
validation_phi = compute_design_matrix(X, centers, spreads)
print(validation_phi)
print(np.shape(validation_phi))

#Closed Form Solution for Validation
w_validation = compute_closed_form_sol(0.1,np.array(validation_phi), np.array(letor_validation_output, dtype = np.float))
print(w_validation)
closed_form_validation = np.matmul(np.transpose(w),np.transpose(phi))
print(closed_form_validation)
print(np.shape(closed_form_validation))

#Calculating the error
N = np.shape(letor_training_output)[0]
D = np.shape(letor_training_output)[1]
#print(N, D)
training_erms = 0
output = np.array(letor_training_output, dtype = np.int)
for i in range(N):
    training_erms +=  ((closed_form[i]-output[i])**2)
training_erms = 0.5 * training_erms
#print(training_erms)
erms = (2 * (training_erms/N))**0.5
print("Training rms: ",erms)

#Error for Validation Set
N = np.shape(letor_validation_output)[0]
D = np.shape(letor_validation_output)[1]
#print(N, D)
validation_err = 0
output = np.array(letor_validation_output, dtype = np.int)
for i in range(N):
    validation_err +=  ((closed_form_validation[i]-output[i])**2)
validation_err = 0.5 * validation_err
#print("Validation Error: ", validation_err)
erms = (2 * validation_err/N)**0.5
print("Validation rms: ", erms)

#SGD

N = np.shape(letor_training_output)[0]
print(SGD_sol(1,N,10000,0.1,phi,letor_training_output))
N = np.shape(letor_validation_output[0])

print(SGD_sol(1,N,10000,0.1,validation_phi,letor_validation_output))


#SYNTHETIC DATASET

#Partitioning
num_rows_syn = np.shape(syn_input_data)[0]
num_columns_syn = np.shape(syn_input_data)[1]

#print(num_rows_syn)
syn_training_input = []
syn_training_output = []
syn_validation_input = []
syn_validation_output = []
syn_testing_input = []
syn_testing_output = []

#Training Set
for i in range(0, int(0.8*num_rows_syn)):
    syn_training_input.append(syn_input_data[i])
    syn_training_output.append(syn_output_data[i])

#Validation Set
for i in range(int(0.8*num_rows_syn), int(0.9*num_rows_syn)):
    syn_validation_input.append(syn_input_data[i])
    syn_validation_output.append(syn_output_data[i])

#Testing Set
for i in range(int(0.9*num_rows_syn), num_rows_syn):
    syn_testing_input.append(syn_input_data[i])
    syn_testing_output.append(syn_output_data[i])

# Hyper parameters and training
X = np.array(syn_training_input)
M = 5
kmeans = KMeans(n_clusters = M, random_state = 0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

clusters_in_label = {i: X[np.where(labels == i)] for i in range(kmeans.n_clusters)}
spreads = list((np.cov(clusters_in_label[i].T)) for i in range(kmeans.n_clusters))
centers = centers[:,np.newaxis, :]

#Design Matrix
phi = compute_design_matrix(X, centers, spreads)
print(phi)
print(np.shape(phi))

#Closed Form Solution
w = compute_closed_form_sol(0.2, np.array(phi), np.array(syn_training_output, dtype=np.float))
print(w)
closed_form = np.matmul(np.transpose(w), np.transpose(phi))
print(closed_form)
print(np.shape(closed_form))

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxVALIDATIONxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Tuning hyper parameters using validation set
X = np.array(syn_validation_input)
M = 5
kmeans = KMeans(n_clusters = M, random_state = 0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
kmeans_list = {j: X[np.where(labels == j)] for j in range(kmeans.n_clusters)}
spreads = list((np.cov(kmeans_list[j].T)) for j in range(kmeans.n_clusters))
centers = centers[:, np.newaxis, :]

#Design Matrix for Validation
validation_phi = compute_design_matrix(X, centers, spreads)
print(validation_phi)
print(np.shape(validation_phi))

#Closed Form Solution for Validation
w_validation = compute_closed_form_sol(0.2,np.array(validation_phi), np.array(syn_validation_output, dtype = np.float))
print(w_validation)
closed_form_validation = np.matmul(np.transpose(w),np.transpose(phi))
print(closed_form_validation)
print(np.shape(closed_form_validation))

#Calculating the error
N = np.shape(syn_training_output)[0]
D = np.shape(syn_training_output)[1]
#print(N, D)
training_erms = 0
output = np.array(syn_training_output, dtype = np.int)
for i in range(N):
    training_erms +=  ((closed_form[i]-output[i])**2)
training_erms = 0.5 * training_erms
#print(training_erms)
erms = (2 * (training_erms/N))**0.5
print("Training rms: ",erms)

#Error for Validation Set
N = np.shape(syn_validation_output)[0]
D = np.shape(syn_validation_output)[1]
#print(N, D)
validation_err = 0
output = np.array(syn_validation_output, dtype = np.int)
for i in range(N):
    validation_err +=  ((closed_form_validation[i]-output[i])**2)
validation_err = 0.5 * validation_err
#print("Validation Error: ", validation_err)
erms = (2 * validation_err/N)**0.5
print("Validation rms: ", erms)


#SGD

N = np.shape(syn_training_output)[0]
print(SGD_sol(1,N,10000,0.1,phi,syn_training_output))
N = np.shape(syn_validation_output[0])
print(SGD_sol(1,N,10000,0.1,validation_phi,syn_validation_output))

