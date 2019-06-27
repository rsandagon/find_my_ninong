const imageUpload = document.getElementById('imageUpload')
const preloader = document.getElementById('preloader')
const uploadContainer = document.getElementById('upload-container')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)


/** Element visibility */
function showElement(el,bol){
  if(bol){
    el.style.display = "block";
  }else{
    el.style.display = "none";
  }
}

async function start() {
  const container = document.createElement('div')
  container.style.position = 'relative'
  document.body.append(container)
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)

  console.log(`labeledFaceDescriptors:${labeledFaceDescriptors}`)

  let image
  let canvas

  showElement(preloader,false);
  showElement(uploadContainer,true);

  imageUpload.addEventListener('change', async () => {
    if (image) image.remove()
    if (canvas) canvas.remove()
    showElement(preloader,true);

    image = await faceapi.bufferToImage(imageUpload.files[0])
    container.append(image)
    canvas = faceapi.createCanvasFromMedia(image)
    container.append(canvas)
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

    showElement(preloader,false);

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas)
    })
  })
}

function loadLabeledImages() {

  // USE THIS TO GET THE DESCRIPTOR OF A TARGET IMAGE
  // I've preloaded the data as Float32Array to reduce http calls

  //const labels = ['yourLabel']
  // return Promise.all(
  //   labels.map(async label => {
  //     const descriptions = []
  //     for (let i = 1; i <= 1; i++) {
  //       const img = await faceapi.fetchImage(`https://m.media-amazon.com/images/M/MV5BYjg5ZWMxY2ItZTc0Yi00MjdlLTkxMTAtZjU4MTE2ODNmMDAwXkEyXkFqcGdeQXVyNTI5NjIyMw@@._V1_UX214_CR0,0,214,317_AL_.jpg`)
  //       const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
  //       descriptions.push(detections.descriptor)
  //     }
  //     return new faceapi.LabeledFaceDescriptors(label, descriptions)
  //   })
  // )

  const label = 'Bayaw'
  descriptions = new Float32Array([-0.1278064250946045,0.11527504771947861,0.04441497474908829,0.01160660944879055,-0.000046702101826667786,-0.10337173938751221,-0.02729238197207451,-0.11031840741634369,0.1702951192855835,-0.052648574113845825,0.2859187424182892,0.06018791347742081,-0.18263399600982666,-0.17679132521152496,0.043665532022714615,0.13464367389678955,-0.13306966423988342,-0.06182605028152466,-0.08608584851026535,-0.007547921501100063,0.054690927267074585,0.0028854478150606155,0.08459945023059845,0.06497157365083694,-0.03674297779798508,-0.3598421812057495,-0.15648393332958221,-0.12691041827201843,0.003876117989420891,-0.07739323377609253,-0.022211488336324692,-0.05685092508792877,-0.22400982677936554,-0.10301804542541504,-0.001941286027431488,-0.02743079513311386,-0.05259840190410614,-0.10022413730621338,0.1969737857580185,-0.016881117597222328,-0.16693395376205444,-0.043886058032512665,0.012508846819400787,0.18302185833454132,0.1447548270225525,0.03344442695379257,0.07049283385276794,-0.10532107949256897,0.06646918505430222,-0.11502932012081146,0.08519822359085083,0.12388107925653458,0.12334716320037842,0.0617288276553154,0.11135561764240265,-0.11514556407928467,0.05598093569278717,0.03276067227125168,-0.14888432621955872,-0.03134584054350853,0.02871587686240673,-0.02608717605471611,-0.03756621479988098,0.016727006062865257,0.2591119408607483,0.11575820297002792,-0.10131874680519104,-0.18946672976016998,0.11367667466402054,-0.10386389493942261,-0.045618847012519836,0.13318851590156555,-0.13000981509685516,-0.18890449404716492,-0.32916778326034546,0.07804044336080551,0.4297143816947937,0.09553107619285583,-0.1961849182844162,0.016707096248865128,-0.204698845744133,-0.04978509992361069,0.07841689884662628,0.08174222707748413,-0.03865843638777733,-0.008827896788716316,-0.13621115684509277,0.013267714530229568,0.1657366305589676,-0.017877664417028427,-0.10571939498186111,0.16728048026561737,-0.020162951201200485,0.02128738909959793,-0.0034588072448968887,0.047477684915065765,-0.06495179235935211,0.035454630851745605,-0.024178845807909966,-0.009937755763530731,0.12384501099586487,0.007522281259298325,-0.03466436266899109,0.11135254800319672,-0.16471824049949646,0.06457024812698364,0.03188179433345795,-0.020224053412675858,0.020061813294887543,0.021809495985507965,-0.16984085738658905,-0.0641758143901825,0.027409281581640244,-0.300121009349823,0.2814030647277832,0.08910124748945236,-0.033637113869190216,0.12485674023628235,0.02699592523276806,0.11470659077167511,-0.055390261113643646,-0.07987156510353088,-0.12720522284507751,-0.07249018549919128,0.08955198526382446,0.07135829329490662,0.11058694124221802,-0.018066853284835815])
  return new faceapi.LabeledFaceDescriptors(label, [descriptions])
}
