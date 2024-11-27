module Models.KNN where
import Models.Types
import Models.LinearModel

import Data.Matrix as M
import Data.Vector as V
import Data.List as D
import Data.Ord (comparing)
import System.Random (randomRIO)
import Control.Parallel.Strategies (parMap, rdeepseq)

{--
K-Nearest Neighbor is a simple classificaiton model that uses no actual learning
The model simply takes an input, sorts known data by which is closest in feature space and classifies input by the highest number of class occurences in k nearest points
E.G: if you have the data set D where a data point has the features x and y and is in the form: ((x, y), classification) and k = 1
    D = {((1,1), dog), ((3,4), cat), ((1,2), dog), ((5,4), cat)} 
    input = (0,0)
    Dsorted = {((1,1), dog), ((1,2), dog), ((3,4), cat), ((5,4), cat)} 
    the nearest point to the input is ((1,1), dog) so input is a dog

The upsides of this model is that the resulting external error is always less then 2 times optimal error
The downsides to this model are that you can never calculate external error and the speed of classification is slow
--}

type Point = [Double]
type Classification = Double
type DataPoint = (Point, Classification)

kNNerror :: Data -> Data -> Int -> Double
kNNerror train test k = 
    let misclassifiedCount = D.foldl (\res (c, d) -> (if (kNN train k d)==c then 0 else 1)+res) 0 (D.zip (M.toList (fst test)) (M.toLists (snd test)))
    in (misclassifiedCount / fromIntegral (D.length (fst test))) * 100


-- Calculate Euclidean distance between two points
euclideanDistance' :: Point -> Point -> Double
euclideanDistance' p1 p2 = sqrt $ D.sum $ D.map (^2) $ D.zipWith (-) p1 p2
-- Sort dataset by distance to the input point
sortDataByDistance :: Point -> [DataPoint] -> [DataPoint]
sortDataByDistance input = D.sortBy (comparing (euclideanDistance' input . fst))

-- Find the most common classification in the k nearest neighbors
mostCommonClassification :: [Classification] -> Classification
mostCommonClassification classes = snd . D.maximum $ D.map (\x -> (D.length x, D.head x)) (group $ sort classes)
  where
    group = D.foldr (\x acc -> if D.null acc || D.head acc /= [x] then [x] : acc else (x : D.head acc) : D.tail acc) []
    sort = D.foldr (\x acc -> if D.null acc || x >= D.head acc then x : acc else acc D.++ [x]) []

-- Classify an input point based on k nearest neighbors
kNN :: Data -> Int -> Point  -> Double
kNN (y, x) k input=
    let dataset = D.zip (M.toLists x) (M.toList y)
        sortedData = D.take k (sortDataByDistance input dataset)
        classifications = D.map snd sortedData
    in mostCommonClassification classifications

{--
Radial Basis Functions Network with Gaussian kernel
--}
rbf::Model
rbf w (y, x) iterations = undefined 

gaussianKernel::Double->Double
gaussianKernel z = exp (-0.5 * (z**2))

norm::Matrix Double->Double
norm x = sqrt $ D.sum $ D.map (^2) $ M.toList x

phi::Matrix Double->Matrix Double->Double->Double
phi x c r = norm (elementwise (-) x c) / r

{--
A very simple heuristic for getting such 
a good partition into M clusters is based on a clustering algorithm we will
discuss later. Here is the main idea. First create a set of M well separated
centers for the clusters; we recommend a simple greedy approach. Start by taking 
an arbitrary data point as a center. Now, as a second center, add the point furthest
from the first center; to get each new center, keep adding the point furthest from
the current set of centers until you have M centers (the distance of a point from
a set is the distance to its nearest neighbor in the set). In the example, the red
point is the first (random) center added; the largest blue point is added next and
so on. This greedy algorithm can be implemented in O(M N d) time using appropriate data 
structures. The partition is inferred from the Voronoi regions of each center: 
all points in a center’s Voronoi region belong to that center’s cluster. 
The center of each cluster is redefined as the average data point in the cluster, 
and its radius is the maximum distance from the center to a point in the cluster

μj = 1/|Sj| ∑ xn
           xn∈Sj

rj = max ‖xn − μj‖
    xn∈Sj


We can improve the clusters by obtaining new Voronoi regions defined by these
new centers μj , and then updating the centers and radii appropriately for these
new clusters. Naturally, we can continue this process iteratively (we introduce this
algorithm later in the chapter as Lloyd’s algorithm for clustering). The main goal
here is computational efficiency, and the perfect partition is not necessary. The
partition, centers and radii after one refinement are illustrated on the right. Note
that though the clusters are disjoint, the spheres enclosing the clusters need not be.
The clusters, together with μj and rj can be computed in O(M N d) time.
If the depth of the partitioning is O(log N ), which is the case if the clusters
are nearly balanced, then the total run time to compute all the clusters at
every level is O(M N d log N ). In Problem 6.16 you can experiment with the
performance gains from such simple branch and bound techniques.


Lloyd’s Algorithm for k-Means Clustering:
1: Initialize μj (e.g. using the greedy approach above).
2: Construct Sj to be all points closest to μj .
3: Update each μj to equal the centroid of Sj .
4: Repeat steps 2 and 3 until Ein stops decreasing.

--}

-- Function to calculate the Euclidean distance between two vectors
euclideanDistance :: V.Vector Double -> V.Vector Double -> Double
euclideanDistance v1 v2 = sqrt $ V.sum $ V.zipWith (\x y -> (x - y) ^ 2) v1 v2

-- Function to initialize cluster centers using a greedy approach
initializeCenters :: Int -> M.Matrix Double -> IO (M.Matrix Double)
initializeCenters m features = do
    let n = M.nrows features
    firstCenterIdx <- randomRIO (1, n)
    let firstCenter = M.getRow firstCenterIdx features
    let centers = V.singleton firstCenter
    initializeCenters' (m - 1) features centers

initializeCenters' :: Int -> M.Matrix Double -> V.Vector (V.Vector Double) -> IO (M.Matrix Double)
initializeCenters' 0 _ centers = return $ fromRows $ V.toList centers
initializeCenters' m features centers = do
    let distances = V.map (\row -> V.minimum $ V.map (euclideanDistance row) centers) (toRows features)
    let maxDistIdx = V.maxIndex distances
    let newCenter = M.getRow (maxDistIdx + 1) features
    initializeCenters' (m - 1) features (V.snoc centers newCenter)

-- Function to assign points to the nearest cluster center
assignClusters :: M.Matrix Double -> M.Matrix Double -> V.Vector Int
assignClusters centers features = V.map (assignCluster centers) (toRows features)

assignCluster :: M.Matrix Double -> V.Vector Double -> Int
assignCluster centers point = V.minIndex $ V.map (euclideanDistance point) (toRows centers)

-- Function to update cluster centers to be the centroid of the points assigned to them
updateCenters :: Int -> M.Matrix Double -> V.Vector Int -> M.Matrix Double
updateCenters m features assignments = fromRows $ D.map (updateCenter features assignments) [0..(m-1)]

updateCenter :: M.Matrix Double -> V.Vector Int -> Int -> V.Vector Double
updateCenter features assignments clusterIdx = V.map (/ count) $ V.foldl1' (V.zipWith (+)) clusterPoints
  where
    clusterPoints = V.map (getRow' features) $ V.elemIndices clusterIdx assignments
    count = fromIntegral $ V.length clusterPoints

getRow' :: M.Matrix Double -> Int -> V.Vector Double
getRow' mat idx = M.getRow (idx + 1) mat

-- Lloyd's Algorithm for k-means clustering
kMeans :: Int -> Data -> IO (M.Matrix Double, V.Vector Int)
kMeans m (labels, features) = do
    centers <- initializeCenters m features
    kMeans' centers features

kMeans' :: M.Matrix Double -> M.Matrix Double -> IO (M.Matrix Double, V.Vector Int)
kMeans' centers features = do
    let assignments = assignClusters centers features
    let newCenters = updateCenters (M.nrows centers) features assignments
    if centers == newCenters
        then return (centers, assignments)
        else kMeans' newCenters features

-- Gaussian RBF kernel function
rbfKernel :: V.Vector Double -> V.Vector Double -> Double -> Double
rbfKernel x c sigma = exp (-0.5 * (euclideanDistance x c) ** 2 / (sigma ** 2))

-- Compute the RBF kernel matrix
computeRBFKernel :: M.Matrix Double -> M.Matrix Double -> Double -> M.Matrix Double
computeRBFKernel centers features sigma = M.fromLists $ parMap rdeepseq (\x -> D.map (\c -> rbfKernel x c sigma) (V.toList $ toRows centers)) (V.toList $ toRows features)

-- Train the RBF network
trainRBF :: M.Matrix Double -> Data -> Double -> Int -> IO (M.Matrix Double)
trainRBF centers (labels, features) sigma iterations = do
    let rbfKernelMatrix = computeRBFKernel centers features sigma

    let pi = pseudoInverse (labels, rbfKernelMatrix) 0.01
    return pi

-- Predict with the RBF network
predictRBF :: (Matrix Double, Matrix Double, Double) -> M.Matrix Double -> M.Matrix Double
predictRBF (centers, weights, sigma) features = computeRBFKernel centers features sigma * weights

-- Radial Basis Functions Network with Gaussian kernel
rbfModel :: Int -> Double -> Data -> Int -> IO (Matrix Double, Matrix Double, Double)
rbfModel m sigma dat@(labels, features) iterations = do
    (centers, _) <- kMeans m dat
    weights <- trainRBF centers dat sigma iterations
    return (centers, weights, sigma)

-- Helper function to compute the pseudoinverse of a matrix
pinv :: M.Matrix Double -> M.Matrix Double
pinv mat = let (u, s, v) = svd mat
               sInv = M.diagonal 0 (V.map (\x -> if x == 0 then 0 else 1 / x) (getDiag s))
           in v * sInv * M.transpose u

-- Singular Value Decomposition (SVD) using Data.Matrix
svd :: M.Matrix Double -> (M.Matrix Double, M.Matrix Double, M.Matrix Double)
svd mat = let (u, s, v) = svdDecompose mat
          in (fromLists u, M.diagonal 0 (V.fromList s), fromLists v)

-- Helper functions to replace missing functions from Data.Matrix
fromRows :: [V.Vector Double] -> M.Matrix Double
fromRows rows = M.fromLists $ D.map V.toList rows

toRows :: M.Matrix Double -> V.Vector (V.Vector Double)
toRows mat = V.fromList $ D.map V.fromList $ M.toLists mat

-- Placeholder for SVD decomposition
svdDecompose :: M.Matrix Double -> ([[Double]], [Double], [[Double]])
svdDecompose mat = 
    let (u, l, p, _) = luDecompUnsafe mat
        s = getDiag l
        v = M.transpose p
    in (M.toLists u, V.toList s, M.toLists v)