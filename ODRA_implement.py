class LoadData:

  def __init__(self,datasetFileName):
    if datasetFileName == 'ionosphere.csv':
      self.df = pd.read_csv('/content/OutlierDetect/Data/ionosphere.csv')
      print(self.df.head())


  def encode(self,features):
    if features == 0:
      print('********************* Before ***********************')
      print()
      print(self.df.head())
      print()
      le = LabelEncoder()
      self.df.iloc[:,-1] = le.fit_transform(self.df.iloc[:,-1])
      print('********************* After ***********************')
      print()
      print(self.df.head())
      print()
      return self.df
    
    elif features == 1:
      st = input('All Columns? (Y/N): ')
      if st == 'Y':
        print('********************* Before ***********************')
        print()
        print(self.df.head())
        print()
        print('********************* After ***********************')
        le = LabelEncoder()
        for i in range(self.df.shape[1]):
          self.df.iloc[:,i] = le.fit_transform(self.df.iloc[:,i])
        
        print()
        print(self.df.head())
        print()
        return self.df

      elif st == 'N':
        listToencode = []

        print('********************* Enter the list of features to encode (enter # to end) ***********************')
        print()
        fts = 0
        while fts !='#':
          fts = input()
          listToencode.append(int(fts))

        print('********************* Before ***********************')
        print()
        print(self.df.head())
        print()
        print('********************* After ***********************')
        le = LabelEncoder()
        for i in listToencode:
          self.df.iloc[:,i] = le.fit_transform(self.df.iloc[:,i])
        
        print()
        print(self.df.head())
        print()
        return self.df




      else:
        print('Invalid input ... Try again!!') 
    
    else:
      print('Invalid input ... Try again!!')



class ODRA:
  def __init__(self,df):
    self.data_df = df
    self.data = np.array(self.data_df.iloc[:,:-1])
    self.y = np.array(self.data_df.iloc[:,-1])
    self.k = len(self.data_df)
    self.sparseMatrix = []

  def distance_oneD(self,i,j):
    return math.fabs(i-j)

  def nn_1d(self,i,j,oneD_array):
    nearest_neighbors = []
    for cnt in range(len(oneD_array)):
      nearest_neighbors.append((self.distance_oneD(oneD_array[i],oneD_array[cnt]),cnt))
    return nearest_neighbors

  def k_plus1_neighbors(self,i,j,oneD_array):
    nearest_neighbors = self.nn_1d(i,j,oneD_array)
    nearest_neighbors = sorted(nearest_neighbors, key=lambda x: x[0])
    return nearest_neighbors[:self.k+1]

  def center_nn(self,nn,oneD_array):
    sum = 0
    for i in nn:
      sum += oneD_array[i[1]]
    return float(sum/(self.k+1))

  def sparseness_degree(self,i,j):
    oneD_array = self.data[:,j]
    #print(oneD_array)
    kplus1_nn = self.k_plus1_neighbors(i,j,oneD_array)
    c_ij = self.center_nn(kplus1_nn,oneD_array)
    sum = 0
    for cnt in kplus1_nn:
      #print(oneD_array[cnt[1]])
      sum += (oneD_array[cnt[1]] - c_ij)**2
    return float(sum/(self.k+1))
    #return c_ij

  def compute_sparseness(self,i):
    sprse = []
    for m in range(self.data.shape[1]):
      sprse.append(self.sparseness_degree(i,m))
    return sprse

  def construct_sparse_matrix(self):
    for n in range(self.data.shape[0]):
      self.sparseMatrix.append(self.compute_sparseness(n))

def main():
    
    data_name = input("Enter Dataset Name: ")
    ld = LoadData(data_name+'.csv')
    df = ld.encode(0)
    od = ODRA(df)
    od.construct_sparse_matrix()
