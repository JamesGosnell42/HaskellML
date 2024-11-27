{-# OPTIONS_GHC -Wall #-}
module Main where
import Models.LinearModel
import Models.LogisticModel
import Models.KNN
import Models.Util
import Models.Types
import DataParser.DataReader

import Control.Concurrent.Async (mapConcurrently)
import Data.Matrix as M
import Data.List as D

scale::Double-> Double
scale x = 2/(sqrt x)


-- Cross-validation function for finding Ecv and weights of a model
crossValidationknn :: Data -> Double -> IO Double
crossValidationknn dat@(y, x) k = do
    errors <- mapM (\i -> runModelknn dat k i) [1, 11..nrows y]
    let avgError = (D.sum errors) / (fromIntegral (D.length errors))
    return avgError

-- Helper function for running model with cross-validation
runModelknn :: Data -> Double -> Int -> IO Double
runModelknn (y, x) k rowidx = do
    let rowend = if (rowidx + 10 > nrows y) then nrows y else rowidx + 10
    let testy = submatrix rowidx rowend 1 (ncols y) y
        testx = submatrix rowidx rowend 1 (ncols x) x
        trainy = removeRows rowidx rowend y
        trainx = removeRows rowidx rowend x
    return $ kNNerror (trainy, trainx) (testy, testx) ((round k) :: Int)

-- Cross-validation function for finding Ecv and weights of a model
crossValidationrbf :: Data -> Double -> IO Double
crossValidationrbf dat@(y, x) k = do
    errors <- mapM (\i -> runModelrbf dat k i) [1, 11..nrows y]
    let avgError = (D.sum errors) / (fromIntegral (D.length errors))
    return avgError

-- Helper function for running model with cross-validation
runModelrbf :: Data -> Double -> Int -> IO Double
runModelrbf (y, x) k rowidx = do
    let rowend = if (rowidx + 10 > nrows y) then nrows y else rowidx + 10
    let testy = submatrix rowidx rowend 1 (ncols y) y
        testx = submatrix rowidx rowend 1 (ncols x) x
        trainy = removeRows rowidx rowend y
        trainx = removeRows rowidx rowend x

    (centers, weights, sigma) <- rbfModel ((round k) :: Int) (scale k) (trainy, trainx) 100
    let preditions = predictRBF (centers, weights, sigma) testx
    let incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) preditions testy
    let misclassifiedCount = sum $ M.toList incorect
    return ((fromIntegral misclassifiedCount / fromIntegral (M.nrows testy)) * 100)


main :: IO ()
main = do
    (datnn, dattest) <- readImagesOneAll "ZipDigitsFull.txt" 300
    printResults datnn "normalizedData.txt"
    printResults dattest "normalizedDataTest.txt"
    --print "KNN"
    --erro1 <- mapM (crossValidationknn datnn) [1..50]
    --print$D.zip [1..50] erro1

    print "RBF"
    (centers, weights, sigma) <- rbfModel 53 (scale 53) datnn 100
    print centers 
    print weights 
    print sigma

{--
    (centers, weights, sigma) <- rbfModel 100 (scale 10) datnn 100
    
    let preditions = predictRBF (centers, weights, sigma) (snd datnn)
    
    let incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) preditions (fst datnn)
    let misclassifiedCount = sum $ M.toList incorect
    print ((fromIntegral misclassifiedCount / fromIntegral (M.nrows (fst datnn))) * 100)
    
    
    let weights = initializeWeights dat8th 0
    print$M.transpose weights
    print "Regression 0 Etest: "
    print (errorCalc weights (snd dattest8th) (fst dattest8th))

    let lambdas = [0, 0.1..4]
    let (cverrors, testerrors) = runLambdas dat8th dattest8th lambdas 
    let lambdaErrorPairs = zip lambdas cverrors
    let (bestLambda, err) = minimumBy (comparing snd) lambdaErrorPairs
    let weightsbest = initializeWeights dat8th bestLambda

    print$M.transpose weightsbest
    print "Lambdas: "
    print lambdas
    print "CV errors: "
    print cverrors
    print "Test errors: "
    print testerrors
    print "Best Lambda: "
    print bestLambda
    print "Best Lambda Ecv: "
    print err
    print "Best Lambda Etest: "
    print (errorCalc weightsbest (snd dattest8th) (fst dattest8th))
    --}