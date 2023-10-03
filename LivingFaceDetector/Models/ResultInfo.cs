namespace LivingFaceDetector.Models
{
    public class ResultInfo
    {
        public string FileName { get; set; }
        public string Result { get; set; }

        public override string ToString()
        {
            return $"{FileName} : {Result}";
        }
    } 
}
