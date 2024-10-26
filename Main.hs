{-# OPTIONS_GHC -Wall #-}
module Main where
import Data.Matrix as M
import Models.Types
import Models.LinearModel
import DataParser.DataReader
import Data.List
main :: IO ()
main =do
    dat <- readImagesOneFive "ZipDigits.txt" -- Extract the value from the IO action
    let datfin = addBiases dat
    --print$head$snd datfin
    --print$pseudoInverse$head$snd datfin
    --print$initializeWeights datfin
    --print$head$fst datfin
    --print$(M.transpose$head$snd datfin)*(initializeWeights datfin)

    let weights = linearRegressionPocket (initializeWeights datfin) datfin 100
    print$initializeWeights datfin
    print (head$fst datfin)
    print (head$snd datfin)
    print (M.toList$M.transpose weights)
    print (M.toList$((M.transpose weights) * (head$snd datfin)))
    print (errorCalc weights (snd datfin) (fst datfin) )
