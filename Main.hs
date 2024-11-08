{-# OPTIONS_GHC -Wall #-}
module Main where
import Data.Matrix as M
import Data.List as D
import Models.LinearModel
import Models.LogisticModel
import DataParser.DataReader

main :: IO ()
main = do


    (datnn, dattest) <- readImagesOneAll "ZipDigitsFull.txt" 300
    let dat8th = polyTransformOrthogonal 8 (normalizeFeatures datnn)
    let dattest8th =polyTransformOrthogonal 8 (normalizeFeatures dattest)
    
    printResults dat8th "resultsFullOrder.txt" 
    printResults dattest8th "resultsFullOrdertest.txt" 

    print ("dat8th row #: " D.++ (show$nrows$snd dat8th) D.++ " dat8th col #: " D.++ (show$ncols$snd dat8th) )
    
    let weightszero = initializeWeights dat8th 0
    let weightsvsmall = initializeWeights dat8th 0.0000000000001
    let weightstwo = initializeWeights dat8th 2

    print$M.transpose weightszero
    print$M.transpose weightsvsmall
    print$M.transpose weightstwo
    {--
    weightsPLA <- pla weights dat 1000
    print weightsPLA
    putStrLn "PLA Ein"
    print (errorCalc weightsPLA (snd dat) (fst dat))
    putStrLn "PLA Eout"
    print (errorCalc weightsPLA (snd dattest) (fst dattest))

    weightsLRP <- linearRegression weights dat 1000
    print weightsLRP
    putStrLn "LRP Ein"
    print (errorCalc weightsLRP (snd dat) (fst dat))
    putStrLn "LRP Eout"
    print (errorCalc weightsLRP (snd dattest) (fst dattest))

    gradientDescentWeights <- gradientDescent weights dat 10
    print gradientDescentWeights

    print (M.multStd (snd dat) gradientDescentWeights)
    print (fmap (\p -> log (1 - exp p)) (M.multStd (snd dat) gradientDescentWeights))
    --Nans from ^
    print (-(D.sum (fmap (\p -> log (1-exp p)) (M.multStd (snd dat) gradientDescentWeights)))) 
    print (fromIntegral (nrows (fst dat)))
    print (-(D.sum (fmap (\p -> log (1-exp p)) (M.multStd (snd dat) gradientDescentWeights))) / (fromIntegral (nrows (fst dat))))

    putStrLn "gradient Descent Ein"
    print (logisticErr gradientDescentWeights (snd dat) (fst dat))
    putStrLn "gradient Descent Eout"
    print (logisticErr gradientDescentWeights (snd dattest) (fst dattest))
    
    let dat3rd = polytransform2to3 dat
    let datest3rd = polytransform2to3 dattest
    let weights3rd = initializeWeights dat3rd
    weightsPLA3rd <- pla weights3rd dat3rd 1000
    print $ M.transpose weightsPLA3rd
    putStrLn "PLA Ein"
    print (errorCalc weightsPLA3rd (snd dat3rd) (fst dat3rd))
    putStrLn "PLA Eout"
    print (errorCalc weightsPLA3rd (snd datest3rd) (fst datest3rd))

    weightsLRP3rd <- linearRegression weights3rd dat3rd 1000
    print $ M.transpose weightsLRP3rd
    putStrLn "LRP Ein"
    print (errorCalc weightsLRP3rd (snd dat3rd) (fst dat3rd))
    putStrLn "LRP Eout"
    print (errorCalc weightsLRP3rd (snd datest3rd) (fst datest3rd))


    gradientDescentWeights3rd <- gradientDescent weights3rd dat3rd 10
    print $ M.transpose gradientDescentWeights3rd
    putStrLn "gradient Descent Ein"
    print (logisticErr gradientDescentWeights3rd (snd dat3rd) (fst dat3rd))
    putStrLn "gradient Descent Eout"
    print (logisticErr gradientDescentWeights3rd (snd datest3rd) (fst datest3rd))
--}