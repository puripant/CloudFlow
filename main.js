// Adapted from https://docs.opencv.org/3.4/db/d7f/tutorial_js_lucas_kanade.html
// SimpleBlobDetector from https://gist.github.com/janpaul123/8b9061d1d093ec0b36dac2230434d34a

function onOpenCvReady() {
  const image_num = 2; //164;
  let images = [];
  let mats = new Array(image_num);
  let seeds = new cv.Mat();

  // parameters for lucas kanade optical flow
  let winSize = new cv.Size(15, 15);
  let maxLevel = 2;
  let criteria = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03);

  let q = d3.queue();
  for (let i = 0; i < image_num; i++) {
    images.push(document.createElement("img"));
    images[i].src = "data/radar" + (i+1) + ".png";
    q.defer(function(callback) {
      images[i].onload = function() {
        mats[i] = new cv.Mat();
        cv.cvtColor(cv.imread(images[i]), mats[i], cv.COLOR_RGB2GRAY)
        // mats[i] = cv.imread(images[i]);

        if (i === 0) { // Blob detection
          let blobs = simpleBlobDetector(mats[i], {
            thresholdStep: 10,
            minThreshold: 100,
            maxThreshold: 1000,
            minRepeatability: 2,
            minDistBetweenBlobs: 100,

            filterByColor: true,
            blobColor: 255,

            filterByArea: true,
            minArea: 2000,
            maxArea: 5000,

            filterByCircularity: true,
            minCircularity: 0.5,
            maxCircularity: 1000000,

            filterByInertia: true,
            minInertiaRatio: 0.1, //0.6,
            maxInertiaRatio: 1000000,

            filterByConvexity: false,
            minConvexity: 0.95, //0.8,
            maxConvexity: 1000000,

            faster: false,
          });

          for (let i = 0; i < blobs.length; i++) {
            seeds.data32F[i*2] = blobs[i].pt.x;
            seeds.data32F[i*2 + 1] = blobs[i].pt.y;
          }

          // for (let blob of blobs) {
          //   let center = new cv.Point(blob.pt.x, blob.pt.y);
          //   cv.circle(mats[i], center, blob.size/2, [255, 0, 0, 255], -1);
          // }
          // cv.imshow("canvas", mats[i]);

        } else { // Optical flow
          let next = new cv.Mat();
          let status = new cv.Mat();
          let err = new cv.Mat();

          // calculate optical flow
          cv.calcOpticalFlowPyrLK(mats[i-1], mats[i], seeds, next, status, err); //, winSize, maxLevel, criteria);

          // select good points
          let goodNew = [];
          let goodOld = [];
          for (let i = 0; i < status.rows; i++) {
            if (status.data[i] === 1) {
              goodNew.push(new cv.Point(next.data32F[i*2], next.data32F[i*2+1]));
              goodOld.push(new cv.Point(seeds.data32F[i*2], seeds.data32F[i*2+1]));
            }
          }

          // draw the tracks
          let zeroEle = new cv.Scalar(0, 0, 0, 255);
          let mask = new cv.Mat(mats[i].rows, mats[i].cols, mats[i].type(), zeroEle);
          for (let i = 0; i < goodNew.length; i++) {
            cv.line(mask, goodNew[i], goodOld[i], [255, 0, 0, 255], 2);
            cv.circle(frame, goodNew[i], 5, [255, 0, 0, 255], -1);
          }
          cv.add(mats[i], mask, mask);

          cv.imshow("canvas", mask);

          next.delete();
          status.delete();
          err.delete();
          mask.delete();
          mats[i-1].delete();
        }

        callback(null);
      }
    });
  }
  q.awaitAll(function(error) {
    if (error) throw error;
    
    mats[image_num-1].delete();
    seeds.delete();

    console.log("all loaded!");
  });
}
