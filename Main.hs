{-# OPTIONS_GHC -Wall #-}
module Main where
import Data.Matrix as M
import Data.List as D
import Data.Ord (comparing)
import Models.LinearModel
import Models.LogisticModel
import Models.Util
import DataParser.DataReader

crossValidationRegression :: Data -> Data -> Double -> (Double, Double)
crossValidationRegression dat@(y, _) (ty, tx) lambda = 
    let errors = map (\i -> runRegression dat lambda i) [1..nrows y]
        avgError = (D.sum errors) / (fromIntegral (D.length errors))
        weights = initializeWeights dat lambda
    in (avgError, errorCalc weights tx ty)

runRegression :: Data -> Double -> Int -> Double
runRegression (y, x) lambda rowidx = 
    let testy = M.rowVector (getRow rowidx y)
        testx = M.rowVector (getRow rowidx x)
        trainy = removeRow rowidx y
        trainx = removeRow rowidx x
        weights = initializeWeights (trainy, trainx) lambda
    in errorCalc weights testx testy

runLambdas:: Data -> Data ->[Double] -> ([Double], [Double])
runLambdas dat dattest lambdas = 
    let errors = map (\l -> crossValidationRegression dat dattest l) lambdas
    in unzip errors

main :: IO ()
main = do

    (datnn, dattest) <- readImagesOneAll "ZipDigitsFull.txt" 300
    let nomdat = normalizeFeatures datnn
    let nomdattest = normalizeFeatures dattest
    printResults nomdat "normalizedData.txt"
    printResults nomdattest "normalizedTestData.txt"
    let dat8th = polyTransformOrthogonal 8 nomdat
    let dattest8th = polyTransformOrthogonal 8  nomdattest
    printResults dat8th "resultsFullOrder.txt"
    printResults dattest8th "resultsFullOrderTest.txt"

    print ("dat8th row #: " ++ (show $ nrows $ snd dat8th) ++ " dat8th col #: " ++ (show $ ncols $ snd dat8th))


    let weights37 = initializeWeights dat8th 0.2
    print$M.transpose weights37
    print "Regression 0.2 Etest: "
    print (errorCalc weights37 (snd dattest8th) (fst dattest8th))
    {--
    
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