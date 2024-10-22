module DataReader.DataReader where
    
import Data.Matrix as M ( Matrix, fromList, getCol, toList )
import qualified Data.Vector as V
import Data.Foldable as F
import Data.Maybe
import System.IO

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
readImages :: FilePath -> IO [(Int, Matrix Double)]
readImages filePath = do
    contents <- readFile filePath
    let linesOfData = lines contents
    return [parseLine line | line <- linesOfData]

-- Helper function that will take in a column and add 1 every time it goes from almost fully white to almost fully black
-- if a pixel is white then it will be -1, if black it will be 1. 
overlaps :: [Double] -> Double
overlaps [] = 0
overlaps [_]  = 0 -- If there's only one pixel left, no more overlaps to check
overlaps (x1:x2:xs)  =
    let isOverlap = (x1 >= 0 && x2 < 0) || (x1 < 0 && x2 >= 0)
    in (if isOverlap then 1 else 0) + overlaps (x2:xs) 

-- Helper Function to compute features for a matrix of pixels
features :: Matrix Double -> Matrix Double
features pixels = 
    let overlapSum = F.sum [overlaps y | z <- [1..16], let y = V.toList $ getCol z pixels]
        intensitySum = F.sum [F.sum y | z <- [1..16], let y = V.toList $ getCol z pixels]
    in M.fromList 2 1 [overlapSum, intensitySum]

-- Helper function to parse each line
parseLine :: String -> (Int, Matrix Double)
parseLine line =
    let values = Prelude.map read $ words line :: [Double]
        number = round (Prelude.head values) :: Int
        pixels = M.fromList 16 16 (Prelude.tail values)
    in (number, features pixels)

parseLineOneFive :: String -> Maybe (Int, Matrix Double)
parseLineOneFive line =
    let values = Prelude.map read $ words line :: [Double]
        number = round (Prelude.head values) :: Int
        pixels = M.fromList 16 16 (Prelude.tail values)
    in case number of
        1 -> Just (number, features pixels)
        5 ->Just (number, features pixels)
        _ -> Nothing

readImagesOneFive :: FilePath -> IO [(Int, Matrix Double)]
readImagesOneFive filePath = do
    contents <- readFile filePath
    let linesOfData = lines contents
    return $ Data.Maybe.mapMaybe parseLineOneFive linesOfData

printResults :: IO ()
printResults = do
    onefive <- readImagesOneFive "ZipDigits.txt"
    let one = [(a, M.toList b) | (a, b) <- onefive, a == 1]
    let five = [(a, M.toList b) | (a, b) <- onefive, a == 5]
    
    let oneResults = unlines (map show one)
    let fiveResults = unlines (map show five)
    
    writeFile "results.txt" (oneResults ++ fiveResults)

