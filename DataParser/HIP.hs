{-# LANGUAGE FlexibleContexts #-}
module DataParser.HIP where

import Graphics.Image as Hip
import Graphics.Image.Interface (Pixel, R, X, Y, Index)
import Data.Array (Array, listArray)
import Data.Array.IArray (IArray, bounds, (!))

-- Function to convert an image to a grayscale array with positions
imageToGrayArray :: FilePath -> IO (Array (Int, Int) Double)
imageToGrayArray path = do
    -- Load the image
    img <- Hip.readImageRGB Hip.VU path
    let grayImg = Hip.map (toGray) img
    let (width, height) = dims grayImg

    -- Map each pixel to its (x, y) position and grayscale value and convert to array
    let pixels = [((x, y), Hip.index grayImg (y, x)) | y <- [0 .. height - 1], x <- [0 .. width - 1]]
        grayscaleValues = map (\((x, y), px) -> ((x, y), px)) pixels
    return $ listArray ((0, 0), (width - 1, height - 1)) (map snd grayscaleValues)

-- Helper function to convert RGB to grayscale
toGray :: Pixel Hip.RGB Double -> Double
toGray = Hip.luminance

testFunc :: IO ()
testFunc = do
    -- Specify your image file path
    let imagePath = "testImage.png"
    grayArray <- imageToGrayArray imagePath
    mapM_ (print . (\ix -> (ix, grayArray ! ix))) (range (bounds grayArray))
