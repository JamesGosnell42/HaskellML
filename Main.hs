{-# OPTIONS_GHC -Wall #-}
module Main where
import Models.LinearModel
import Models.LogisticModel
import Models.Util
import Models.Types
import DataParser.DataReader

import Data.Matrix as M
import Data.List as D



crossValidationRegression :: Data -> Data -> Double -> (Double, Double)
crossValidationRegression dat@(y, _) (ty, tx) lambda = 
    let errors = map (\i -> runRegression dat lambda i) [1..nrows y]
        avgError = (D.sum errors) / (fromIntegral (D.length errors))
        weights = pseudoInverse dat lambda
    in (avgError, errorCalc weights (ty, tx))

runRegression :: Data -> Double -> Int -> Double
runRegression (y, x) lambda rowidx = 
    let testy = M.rowVector (getRow rowidx y)
        testx = M.rowVector (getRow rowidx x)
        trainy = removeRow rowidx y
        trainx = removeRow rowidx x
        weights = pseudoInverse (trainy, trainx) lambda
    in errorCalc weights (testy, testx)

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

    print "pseudoInverse 0.01"
    let weights37 = pseudoInverse nomdat 0.01
    weightsfin <- pla weights37 nomdat 100
    print$M.transpose weightsfin
    print (errorCalc weightsfin nomdat)
    print (errorCalc weightsfin nomdattest)
    
    print "linearRegression 100 itr"
    weightsfin2 <- linearRegression weights37 nomdat 100
    print$M.transpose weightsfin2
    print (errorCalc weightsfin2 nomdat)
    print (errorCalc weightsfin2 nomdattest)

    print "gradientDescent 10 itr"
    weightsfin3 <- gradientDescent weights37 nomdat 10
    print$M.transpose weightsfin3

    print (logisticErr weightsfin3 nomdat)
    print (logisticErr weightsfin3 nomdattest)

    print "stochasticGradientDescent 100 itr"
    weightsfin4 <- stochasticGradientDescent weights37 nomdat 100
    print$M.transpose weightsfin4
    print (logisticErr weightsfin4 nomdat)
    print (logisticErr weightsfin4 nomdattest)

    print "pseudoInverse 8th 0"
    let weights = pseudoInverse dat8th 0
    print$M.transpose weights
    print (errorCalc weights dat8th)
    print (errorCalc weights dattest8th)

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