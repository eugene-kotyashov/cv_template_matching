#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int main() {
    // Load images
    cv::Mat templateImg = cv::imread("template_image.jpeg");
    cv::Mat largeImg = cv::imread("large_image.jpeg");

    // Convert to grayscale
    cv::Mat templateImgBW, largeImgBW;
    cv::cvtColor(templateImg, templateImgBW, cv::COLOR_BGR2GRAY);
    cv::cvtColor(largeImg, largeImgBW, cv::COLOR_BGR2GRAY);

    // Initialize ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> templateKp, largeKp;
    cv::Mat templateDes, largeDes;
    orb->detectAndCompute(templateImgBW, cv::Mat(), templateKp, templateDes);
    orb->detectAndCompute(largeImgBW, cv::Mat(), largeKp, largeDes);

    // Create Brute Force matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    matcher.match(templateDes, largeDes, matches);

    // Sort matches by distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });
    // Estimate homography using RANSAC
    std::vector<cv::Point2f> srcPts, dstPts;
    for (const cv::DMatch& m : matches) {
        srcPts.push_back(templateKp[m.queryIdx].pt);
        dstPts.push_back(largeKp[m.trainIdx].pt);
    }

    std::vector<unsigned char> mask;
    cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, 3.0, mask);

    // Separate good and bad matches
    std::vector<cv::DMatch> goodMatches;
    for (int i = 0; i < matches.size(); ++i) {
        if (mask[i] > 0) {
            goodMatches.push_back(matches[i]);
        }
    }
    // Draw matches
    cv::Mat matchImg;

    cv::drawMatches(
        templateImg, templateKp,
        largeImg, largeKp,
        goodMatches, matchImg,
        cv::Scalar(0, 255, 0), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    // Draw template image's ROI on top of large image
    cv::Mat largeImgClone = largeImg.clone();
    cv::Mat templateImgROI = templateImg.clone();
    std::vector<cv::Point2f> templateCorners(4);
    templateCorners[0] = cv::Point2f(0, 0);
    templateCorners[1] = cv::Point2f(templateImg.cols, 0);
    templateCorners[2] = cv::Point2f(templateImg.cols, templateImg.rows);
    templateCorners[3] = cv::Point2f(0, templateImg.rows);
    std::vector<cv::Point2f> largeCorners(4);
    cv::perspectiveTransform(templateCorners, largeCorners, H);
    cv::line(largeImgClone, largeCorners[0], largeCorners[1], cv::Scalar(0, 255, 0), 2);
    cv::line(largeImgClone, largeCorners[1], largeCorners[2], cv::Scalar(0, 255, 0), 2);
    cv::line(largeImgClone, largeCorners[2], largeCorners[3], cv::Scalar(0, 255, 0), 2);
    cv::line(largeImgClone, largeCorners[3], largeCorners[0], cv::Scalar(0, 255, 0), 2);

    cv::imshow("Template ROI on Large Image", largeImgClone);
    // Display the result
    cv::imshow("Matched Image", matchImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
