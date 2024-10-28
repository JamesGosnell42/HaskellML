{-# OPTIONS_GHC -Wall #-}
module Main where
import Data.Matrix as M
import Models.Types
import Models.LinearModel
import DataParser.DataReader
import Data.List

main :: IO ()
main = do
    dat <- readImagesOneFive "ZipDigits.txt" -- Extract the value from the IO action
    let datfin = addBiases dat
    let weights = initializeWeights datfin

    -- Run the linear regression pocket and regular linear regression
    weightsLRP <- linearRegressionPocket weights datfin 1000
    weightsLR <- linearRegression weights datfin 1000
    weightsGD <- gradientDescent weights datfin 1000
    weightsSGD <- stochasticGradientDescent weights datfin 1000

    -- Print results for linear regression without pocket
    putStrLn "Linear Regression no inter"
    print (M.transpose weights)
    print (errorCalc weights (snd datfin) (fst datfin))

    -- Print results for linear regression with pocket
    putStrLn "Linear Regression pocket"
    print (M.transpose weightsLRP)
    print (errorCalc weightsLRP (snd datfin) (fst datfin))

    -- Print results for regular linear regression
    putStrLn "Linear Regression"
    print (M.transpose weightsLR)
    print (errorCalc weightsLR (snd datfin) (fst datfin))

    -- Print results for regular gradient Descent
    putStrLn "gradient Descent"
    print (M.transpose weightsGD)
    print (errorCalc weightsGD (snd datfin) (fst datfin))

    putStrLn "Stochastic gradient Descent"
    print (M.transpose weightsSGD)
    print (errorCalc weightsSGD (snd datfin) (fst datfin))