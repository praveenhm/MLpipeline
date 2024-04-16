from sentence_transformers import SentenceTransformer
import scipy.spatial as ssp
  
model = SentenceTransformer("all-mpnet-base-v2")
A = model.encode(['chocolate chip cookies','PLS6;YJBXSRF&/'])

CosineDistance = ssp.distance.cosine(A[0],A[1])
print(CosineDistance)


