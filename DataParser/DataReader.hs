module DataParser.DataReader where
import Models.Util 
import Models.Types

import Data.Matrix as M 
import qualified Data.Vector as V
import Data.List as D
import Data.Foldable as F

import Data.Maybe
import System.IO
import System.Random (randomRIO)
import Control.Parallel.Strategies

{--
Normalized handwritten digits, automatically
scanned from envelopes by the U.S. Postal Service. The original
scanned digits are binary and of different sizes and orientations; the
images  here have been deslanted and size normalized, resulting
in 16 x 16 grayscale images (Le Cun et al., 1990).

The data are in two gzipped files, and each line consists of the digit
id (0-9) followed by the 256 grayscale values.

There are 7291 training observations and 2007 test observations,
distributed as follows:
         0    1   2   3   4   5   6   7   8   9 Total
Train 1194 1005 731 658 652 556 664 645 542 644 7291
 Test  359  264 198 166 200 160 170 147 166 177 2007

or as proportions:
         0    1   2    3    4    5    6    7    8    9 
Train 0.16 0.14 0.1 0.09 0.09 0.08 0.09 0.09 0.07 0.09
 Test 0.18 0.13 0.1 0.08 0.10 0.08 0.08 0.07 0.08 0.09


Alternatively, the training data are available as separate files per
digit (and hence without the digit identifier in each row)

The test set is notoriously "difficult", and a 2.5% error rate is
excellent. These data were kindly made available by the neural network
group at AT&T research labs (thanks to Yann Le Cunn).

--}

-- Function to read the file and parse the data
readImages :: FilePath -> IO (Matrix Double, Matrix Double)
readImages filePath = do
    contents <- readFile filePath
    let linesOfData = lines contents
        parsedData = [parseLine line | line <- linesOfData]
        (labels, matrices) = unzip parsedData
    return (fromLists labels, fromLists matrices)
    
-- Helper function that will take in a column and add 1 every time it goes from almost fully white to almost fully black
-- if a pixel is white then it will be -1, if black it will be 1. 
overlaps :: [Double] -> Double
overlaps [] = 0
overlaps [_]  = 0 -- If there's only one pixel left, no more overlaps to check
overlaps (x1:x2:xs)  =
    let isOverlap = (x1 >= 0 && x2 < 0) || (x1 < 0 && x2 >= 0)
    in (if isOverlap then 1 else 0) + overlaps (x2:xs) 

-- Helper Function to compute features for a matrix of pixels
features :: Matrix Double -> [Double]
features pixels = 
    let overlapSum = F.sum [overlaps y | z <- [1..16], let y = V.toList $ getCol z pixels]
        intensitySum = F.sum [F.sum y | z <- [1..16], let y = V.toList $ getCol z pixels]
    in  [1, overlapSum, intensitySum] -- 1 is the bias weight.

-- Helper function to parse each line
parseLine :: String -> ([Double], [Double])
parseLine line =
    let values = Prelude.map read $ words line :: [Double]
        number = Prelude.head values
        pixels = M.fromList 16 16 (Prelude.tail values)
    in ([number], features pixels)



-- function to read in a line and return the features and the label, +1 for ones and -1 for all other numbers
parseLineOneAll :: String -> Maybe ([Double], [Double])
parseLineOneAll line =
    let values = Prelude.map read $ words line :: [Double]
        number = Prelude.head values
        pixels = M.fromList 16 16 (Prelude.tail values)
    in case number of
        1 -> Just ([1], features pixels)
        _ ->Just ([-1], features pixels)
        
-- Function to read the file and parse the data into a test set and a training set, with a sample size samplen for the training set and labels of 1 for ones and -1 for all other numbers
readImagesOneAll :: FilePath -> Int -> IO ((Matrix Double, Matrix Double), (Matrix Double, Matrix Double))
readImagesOneAll filePath samplen = do
    contents <- readFile filePath
    let linesOfData = lines contents
        totalLines = length linesOfData

    -- Generate unique random indices
    randomIndices <- generateUniqueRandomIndices samplen totalLines

    -- Select sample lines based on random indices
    let selectedLines = [linesOfData !! i | i <- randomIndices]
        remainingIndices = [0..totalLines-1] \\ randomIndices
        remainingLines = [linesOfData !! i | i <- remainingIndices]

    -- Parse the data
    let parsedSampleData = Data.Maybe.mapMaybe parseLineOneAll selectedLines
        parsedRemainingData = Data.Maybe.mapMaybe parseLineOneAll remainingLines
        (sampleLabels, sampleMatrices) = unzip parsedSampleData
        (remainingLabels, remainingMatrices) = unzip parsedRemainingData

    return ((fromLists sampleLabels, fromLists sampleMatrices), (fromLists remainingLabels, fromLists remainingMatrices))



-- Generate a list of unique random indices
generateUniqueRandomIndices :: Int -> Int -> IO [Int]
generateUniqueRandomIndices n maxIndex = do
    generate n []
  where
    generate 0 acc = return acc
    generate n acc = do
        idx <- randomRIO (0, maxIndex - 1)
        if idx `elem` acc
            then generate n acc
            else generate (n - 1) (idx : acc)


--print the results of the parse into a file
printResults ::  (Matrix Double, Matrix Double) -> FilePath ->IO ()
printResults (labels, matrices) outfilePath = do 
    let one = [(a, b) | (a, b) <- zip (M.toList labels) (M.toLists matrices), a == 1]
    let other = [(a, b) | (a, b) <- zip (M.toList labels) (M.toLists matrices), a == -1]
    
    let oneResults = unlines (map show one)
    let otherResults = unlines (map show other)
    
    writeFile outfilePath (oneResults ++ otherResults)


--function to normalize the features of a matrix
normalizeFeatures :: (Matrix Double, Matrix Double) -> (Matrix Double, Matrix Double)
normalizeFeatures (labels, features) =
    let numCols = ncols features
        normalizedFeatures = foldl normalizeColumn features [2..numCols] -- Start from the second column
    in (labels, normalizedFeatures)
  where
    normalizeColumn :: Matrix Double -> Int -> Matrix Double
    normalizeColumn mat colIdx =
        let col = getCol colIdx mat
            minVal = minimum col
            maxVal = maximum col
            shift = (maxVal + minVal) / 2
            scale = (maxVal - minVal) / 2
            normalizeValue x = (x - shift) / scale
        in mapCol (\_ x -> normalizeValue x) colIdx mat

-- Calculate the k-th order Legendre polynomial at x
legendre :: Int -> Double -> Double
legendre 0 _ = 1
legendre 1 x = x
legendre k x = ((2 * fromIntegral k - 1) * x * legendre (k - 1) x - (fromIntegral k - 1) * legendre (k - 2) x) / fromIntegral k

-- Orthogonal polynomial transform using Legendre polynomials
polyTransformOrthogonal :: Int -> (Matrix Double, Matrix Double) -> (Matrix Double, Matrix Double)
polyTransformOrthogonal degree (ys, xs) = 
    let (nys, nxs) = polyTransformOrthogonalList degree (toLists ys) (toLists xs)
    in (fromLists nys, fromLists nxs)

polyTransformOrthogonalList :: Int -> [[Double]] -> [[Double]] -> ([[Double]], [[Double]])
polyTransformOrthogonalList _ [] _ = ([], [])
polyTransformOrthogonalList _ _ [] = ([], [])
polyTransformOrthogonalList degree (y:ys) (x:xs) = case x of
    (1:x1:x2:[]) -> 
        let (yn, xn) = polyTransformOrthogonalList degree ys xs 
            transformed = orthogonalFeatures degree x1 x2
        in (y:yn, transformed:xn)
    _ -> error "Expected exactly two features with a bias variable"

-- Generate orthogonal features using Legendre polynomials up to the given degree
orthogonalFeatures :: Int -> Double -> Double -> [Double]
orthogonalFeatures degree x1 x2 = concatMap (\d -> legendreFeatures d x1 x2) [0..degree]

-- Generate Legendre polynomial features for a given degree
legendreFeatures :: Int -> Double -> Double -> [Double]
legendreFeatures 0 _ _ = [1]
legendreFeatures 1 x1 x2 = [legendre 1 x1, legendre 1 x2]
legendreFeatures d x1 x2 = 
    let l1 = legendre d x1
        l2 = legendre d x2
        crossTerms = [legendre i x1 * legendre (d - i) x2 | i <- [1..(d-1)]]
    in l1 : l2 : crossTerms