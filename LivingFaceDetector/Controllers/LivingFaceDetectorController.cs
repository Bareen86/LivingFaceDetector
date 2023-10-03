using Emgu.CV;
using Microsoft.AspNetCore.Mvc;
using System.Drawing;
using LivingFaceDetector.Extensions;
using Emgu.CV.CvEnum;
using CsvHelper.Configuration;
using CsvHelper;
using LivingFaceDetector.Models;
using System.Globalization;

namespace LivingFaceDetector.Controllers
{
    [Route("api/images")]
    [ApiController]
    public class LivingFaceDetectorController : ControllerBase
    {
        private string livingFacesPath = @"validation_data\living";
        private string fakeFacesPath = @"validation_data\fake";

        [HttpPost("[action]")]
        public async Task<ResultInfo> RecognizeFace(IFormFile imageData)
        {
            try
            {
                using (Mat image = new Mat())
                {
                    CvInvoke.Imdecode(await imageData.GetBytes(), ImreadModes.Color, image);
                    using (CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_alt2.xml"))
                    {
                        Rectangle[] faces = faceCascade.DetectMultiScale(image, 1.1, 1, new Size(20,20));
                        if (faces.Length > 0)
                        {
                            var firstFace = faces[0];
                            Mat firstFaceImage = new Mat(image, firstFace);

                            if (HasEyes(firstFaceImage)) 
                            {
                                if (HasTexture(firstFaceImage))
                                {
                                    return GetInfoString("real", imageData.FileName);
                                }
                                else
                                {
                                    return GetInfoString("fake", imageData.FileName);
                                }
                               
                            }
                            else
                            {
                                return GetInfoString("fake", imageData.FileName);
                            }
                        }
                        else
                        {
                            return GetInfoString("empty", imageData.FileName);
                        }
                    }
                }
            }
            catch 
            {
                return GetInfoString("empty", imageData.FileName);
            }
        }
        public static bool HasTexture(Mat image)
        {
            
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            HOGDescriptor hog = new HOGDescriptor(
                new Size(64, 128),
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9);
            float[] descriptors = hog.Compute(grayImage);
            double textureThreshold = 0.123;
            double hogMean = descriptors.Average();
            return hogMean > textureThreshold;
        }

        public static bool HasEyes(Mat image)
        {

            string eyeCascadePath = "haarcascade_eye.xml";
            CascadeClassifier eyeCascade = new CascadeClassifier(eyeCascadePath);
            double scaleFactor = 1.1;
            int minNeighbors = 7;
            Size minSize = new Size(10, 10);
            Rectangle[] eyes = eyeCascade.DetectMultiScale(image, scaleFactor, minNeighbors, minSize);

            return eyes.Length > 0;
        }
      
        [HttpGet("[action]")]
        public async Task<IActionResult> CheckFaces()
        {
            var allImages = Directory.GetFiles(livingFacesPath, "*.*", SearchOption.AllDirectories)
        .ToList();

            List<ResultInfo> results = new List<ResultInfo>();
            foreach (var imagePath in allImages)
            {
                using (var stream = System.IO.File.OpenRead(imagePath))
                {
                    var i = new Microsoft.AspNetCore.Http.FormFile(stream, 0, stream.Length, null, Path.GetFileName(stream.Name));
                    var result = await RecognizeFace(i);

                    results.Add(new ResultInfo
                    {
                        FileName = result.FileName,
                        Result = result.Result
                    });

                    Console.WriteLine(result);
                }
            }

            Console.WriteLine("Человек: " + results.Count(r => r.Result.Contains("real")));
            Console.WriteLine("Лицо не обнаружено: " + results.Count(r => r.Result.Contains("empty")));
            Console.WriteLine("Фотография: " + results.Count(r => r.Result.Contains("fake")));

            using (var originalMemoryStream = new MemoryStream())
            {
                using (var writer = new StreamWriter(originalMemoryStream))
                using (var csv = new CsvWriter(writer, new CsvConfiguration(CultureInfo.InvariantCulture)))
                {
                    csv.WriteRecords(results);
                }
                var newMemoryStream = new MemoryStream(originalMemoryStream.ToArray());
                newMemoryStream.Seek(0, SeekOrigin.Begin);
                return File(newMemoryStream, "text/csv", "BIClear_preds.csv");
            }
        }

        private ResultInfo GetInfoString(string verdict, string fileName)
        {
            var result = new ResultInfo
            {
                FileName = Path.GetFileName(fileName),
                Result = verdict
            };
            return result;
        }

    }
}
    

    