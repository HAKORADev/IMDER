#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QFrame>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QMenu>
#include <QtWidgets/QAction>
#include <QtWidgets/QSlider>
#include <QtWidgets/QColorDialog>
#include <QtWidgets/QInputDialog>
#include <QtCore/QThread>
#include <QtCore/QTimer>
#include <QtCore/QMutex>
#include <QtCore/QDateTime>
#include <QtCore/QFileInfo>
#include <QtCore/QFile>
#include <QtCore/QDir>
#include <QtGui/QColor>
#include <QtCore/QPoint>
#include <QtCore/QSize>
#include <QtGui/QImage>
#include <QtGui/QPixmap>
#include <QtGui/QIcon>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtGui/QMouseEvent>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>

static QString mainBtnStyle(){return R"(
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #121212,stop:0.3 #121212,stop:0.7 #1a1a1a,stop:1 #121212);
        border:2px solid #E5E5E5;border-radius:8px;font-size:14px;
        font-weight:bold;color:white;padding:8px 16px;}
    QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #121212,stop:0.3 #161616,stop:0.7 #1e1e1e,stop:1 #121212);
        border:2px solid #E5E5E5;}
    QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #0e0e0e,stop:0.3 #121212,stop:0.7 #161616,stop:1 #0e0e0e);
        border:2px solid #E5E5E5;}
    QPushButton:disabled{background-color:#2a2a2a;border:2px solid #555;color:#666;})"; }

    static QString surfBtnStyle(){return R"(
    QPushButton{background-color:#2a2a2a;color:white;border:1px solid #3a3a3a;
        border-radius:5px;font-size:12px;padding:6px 12px;}
    QPushButton:hover{background-color:#3a3a3a;border:1px solid #E5E5E5;}
    QPushButton:pressed{background-color:#4a4a4a;border:1px solid #E5E5E5;}
    QPushButton:disabled{background-color:#2a2a2a;border:1px solid #404040;color:#666;})"; }

    static QString panelStyle(){return
        "QFrame{background-color:#121212;border:2px solid #E5E5E5;border-radius:8px;}";}

        static QString comboStyle(){return R"(
    QComboBox{background-color:#1a1a1a;color:#E5E5E5;border:2px solid #E5E5E5;
        border-radius:6px;padding:6px 12px;min-width:120px;font-size:13px;}
    QComboBox:hover{border:2px solid #E5E5E5;}
    QComboBox:disabled{background-color:#2a2a2a;border:2px solid #555;color:#666;}
    QComboBox::drop-down{border:none;width:24px;}
    QComboBox QAbstractItemView{background-color:#1a1a1a;color:#E5E5E5;
        border:1px solid #E5E5E5;selection-background-color:#2a2a2a;})"; }

        static QString progressStyle(){return R"(
    QProgressBar{border:1px solid #404040;background-color:#1a1a1a;height:8px;
        border-radius:4px;text-align:center;color:#A0A0A0;}
    QProgressBar::chunk{background-color:#A0A0A0;border-radius:3px;})"; }

    static QString menuStyle(){return R"(
    QMenu{background-color:#1a1a1a;color:#E5E5E5;border:1px solid #E5E5E5;
        border-radius:4px;padding:4px;}
    QMenu::item{padding:6px 16px;border-radius:2px;}
    QMenu::item:selected{background-color:#2a2a2a;})"; }

    static QString sliderStyle(){return R"(
    QSlider::groove:horizontal{height:6px;background:#2a2a2a;border-radius:3px;}
    QSlider::handle:horizontal{background:#4CAF50;width:16px;height:16px;
        margin:-5px 0;border-radius:8px;})"; }

        static QString titleLblStyle() { return "color:#E5E5E5;font-weight:bold;font-size:16px;"; }
        static QString subtitleLblStyle(){ return "color:#A0A0A0;font-size:12px;"; }
        static QString windowStyle()    { return "background-color:#0A0A0A;color:#E5E5E5;"; }
        static QString previewLblStyle(){ return "background-color:#1a1a1a;border:1px solid #404040;border-radius:4px;"; }

        static const uint32_t SHA_K[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

            static void sha256Block(uint32_t s[8],const uint8_t b[64]){
                uint32_t w[64],a,b2,c,d,e,f,g,h,t1,t2;
                for(int i=0;i<16;i++) w[i]=((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|((uint32_t)b[i*4+2]<<8)|b[i*4+3];
                for(int i=16;i<64;i++){
                    uint32_t s0=(w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
                    uint32_t s1=(w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
                    w[i]=w[i-16]+s0+w[i-7]+s1;
                }
                a=s[0];b2=s[1];c=s[2];d=s[3];e=s[4];f=s[5];g=s[6];h=s[7];
                for(int i=0;i<64;i++){
                    uint32_t S1=(e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
                    t1=h+S1+((e&f)^(~e&g))+SHA_K[i]+w[i];
                    uint32_t S0=(a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
                    t2=S0+((a&b2)^(a&c)^(b2&c));
                    h=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
                }
                s[0]+=a;s[1]+=b2;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
            }
            static std::string sha256hex(const uint8_t* data,size_t len){
                uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
                uint8_t blk[64]; size_t i=0;
                while(i+64<=len){sha256Block(s,data+i);i+=64;}
                size_t rem=len-i; memcpy(blk,data+i,rem); blk[rem]=0x80;
                if(rem<56) memset(blk+rem+1,0,55-rem);
                else{memset(blk+rem+1,0,63-rem);sha256Block(s,blk);memset(blk,0,56);}
                uint64_t bits=(uint64_t)len*8;
                for(int j=0;j<8;j++) blk[56+j]=(uint8_t)(bits>>(56-j*8));
                sha256Block(s,blk);
                char hex[65]; for(int j=0;j<32;j++) snprintf(hex+j*2,3,"%02x",(s[j/4]>>(24-(j%4)*8))&0xFF);
                return std::string(hex,64);
            }

            static bool writeWav(const std::string& path,const std::vector<int16_t>& s,int sr=44100){
                FILE* f=fopen(path.c_str(),"wb"); if(!f) return false;
                uint32_t ds=(uint32_t)(s.size()*2),rs=36+ds;
                fwrite("RIFF",1,4,f);fwrite(&rs,4,1,f);fwrite("WAVE",1,4,f);
                fwrite("fmt ",1,4,f);uint32_t fs=16;fwrite(&fs,4,1,f);
                uint16_t af=1,ch=1;fwrite(&af,2,1,f);fwrite(&ch,2,1,f);
                uint32_t sr2=(uint32_t)sr;fwrite(&sr2,4,1,f);
                uint32_t br=(uint32_t)sr*2;fwrite(&br,4,1,f);
                uint16_t ba=2,bps=16;fwrite(&ba,2,1,f);fwrite(&bps,2,1,f);
                fwrite("data",1,4,f);fwrite(&ds,4,1,f);fwrite(s.data(),2,s.size(),f);
                fclose(f);return true;
            }

            static bool checkFfmpeg(){return system("ffmpeg -version >/dev/null 2>&1")==0;}

            static bool extractAudio(const std::string& vid,const std::string& out,double dur,int quality){
                if(!checkFfmpeg()){fprintf(stderr,"Error: ffmpeg not found.\n");return false;}
                const char* qmap[]={"32k","64k","96k","128k","160k","192k","224k","256k","320k","copy"};
                int qi=quality/10-1; if(qi<0)qi=0; if(qi>9)qi=9;
                std::string cmd="ffmpeg -i \""+vid+"\"";
                if(dur>0) cmd+=" -t "+std::to_string(dur);
                if(std::string(qmap[qi])=="copy") cmd+=" -c:a copy";
                else cmd+=" -b:a "+std::string(qmap[qi]);
                cmd+=" -y \""+out+"\" >/dev/null 2>&1";
                return system(cmd.c_str())==0;
            }

            static void genSoundForFrame(const cv::Mat& rgbFrame,double frameDur,
                                         int sampleRate,std::vector<int16_t>& out){
                std::string hex=sha256hex(rgbFrame.data,(size_t)rgbFrame.total()*3);
                float freqs[3],amps[3];
                for(int i=0;i<3;i++){
                    unsigned fs=0,as=0;
                    sscanf(hex.c_str()+i*4,"%4x",&fs);
                    sscanf(hex.c_str()+(i+3)*4,"%4x",&as);
                    freqs[i]=50.0f+(float)(fs%4000);
                    amps[i] =0.1f +(float)(as%9000)/10000.0f;
                }
                int n=(int)(sampleRate*frameDur);
                std::vector<float> wave(n,0.f);
                for(int i=0;i<3;i++)
                    for(int k=0;k<n;k++)
                        wave[k]+=amps[i]*sinf(2.f*(float)M_PI*freqs[i]*k/(float)sampleRate);
                float mx=0.f; for(float v:wave) mx=std::max(mx,fabsf(v));
                for(int k=0;k<n;k++){
                    float v=mx>0?wave[k]/mx:wave[k];
                    out.push_back((int16_t)(v*32767.f));
                }
                                         }

                                         static std::string addAudioToVideo(const std::string& videoPath,
                                                                            const std::vector<cv::Mat>& frames,
                                                                            double fps,const std::string& outPath,
                                                                            const std::string& soundOpt,
                                                                            const std::string& targetAudioPath,
                                                                            int audioQuality){
                                             if(soundOpt=="mute") return videoPath;
                                             if(!checkFfmpeg()){fprintf(stderr,"Error: ffmpeg not found.\n");return videoPath;}

                                             char tmpDir[]=("/tmp/imder_XXXXXX");
                                             if(!mkdtemp(tmpDir)) return videoPath;
                                             std::string audioPath=std::string(tmpDir)+"/audio.mp3";

                                             if(soundOpt=="target-sound"&&!targetAudioPath.empty()){
                                                 double dur=frames.empty()?0.0:(double)frames.size()/fps;
                                                 if(!extractAudio(targetAudioPath,audioPath,dur,audioQuality)) return videoPath;
                                             } else if(soundOpt=="sound"){
                                                 int sr=44100; double fd=1.0/fps;
                                                 std::vector<int16_t> full;
                                                 full.reserve(frames.size()*(size_t)(sr*fd+1));
                                                 for(int i=0;i<(int)frames.size();i++){
                                                     genSoundForFrame(frames[i],fd,sr,full);
                                                 }
                                                 std::string wp=std::string(tmpDir)+"/audio.wav";
                                                 audioPath=wp;
                                                 if(!writeWav(audioPath,full,sr)) return videoPath;
                                             } else { return videoPath; }

                                             std::string cmd="ffmpeg -i \""+videoPath+"\" -i \""+audioPath+
                                             "\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y \""+
                                             outPath+"\" >/dev/null 2>&1";
                                             if(system(cmd.c_str())!=0){fprintf(stderr,"Error adding audio.\n");return videoPath;}
                                             (std::string("rm -rf \"")+tmpDir+"\"") ; system(("rm -rf \""+std::string(tmpDir)+"\"").c_str());
                                             return outPath;
                                                                            }

                                                                            namespace GIF {
                                                                                // GIF encoder using ffmpeg - reliable and produces valid GIFs
                                                                                struct Encoder {
                                                                                    std::string outputPath;
                                                                                    std::string tmpDir;
                                                                                    std::vector<cv::Mat> frames;
                                                                                    int w, h, delayMs;

                                                                                    bool open(const std::string& path, int W, int H, int delay_ms) {
                                                                                        w = W; h = H; delayMs = delay_ms;
                                                                                        outputPath = path;
                                                                                        frames.clear();

                                                                                        // Create temp directory for frames
                                                                                        char tmp[] = "/tmp/imder_gif_XXXXXX";
                                                                                        if (!mkdtemp(tmp)) return false;
                                                                                        tmpDir = tmp;
                                                                                        return true;
                                                                                    }

                                                                                    void writeFrame(const cv::Mat& rgb) {
                                                                                        frames.push_back(rgb.clone());
                                                                                    }

                                                                                    void close() {
                                                                                        if (frames.empty()) return;

                                                                                        // Write frames as PNG files
                                                                                        for (size_t i = 0; i < frames.size(); i++) {
                                                                                            cv::Mat bgr;
                                                                                            cv::cvtColor(frames[i], bgr, cv::COLOR_RGB2BGR);
                                                                                            std::string framePath = tmpDir + "/frame_" + std::to_string(i) + ".png";
                                                                                            cv::imwrite(framePath, bgr);
                                                                                        }

                                                                                        // Calculate framerate from delay (delay is in centiseconds)
                                                                                        // delayMs is in milliseconds, convert to fps
                                                                                        double fps = 1000.0 / std::max(10, delayMs);

                                                                                        // Use ffmpeg to create GIF
                                                                                        std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) +
                                                                                        " -i " + tmpDir + "/frame_%d.png" +
                                                                                        " -vf \"scale=" + std::to_string(w) + ":" + std::to_string(h) + ":flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\"" +
                                                                                        " -loop 0 \"" + outputPath + "\" >/dev/null 2>&1";

                                                                                        system(cmd.c_str());

                                                                                        // Cleanup temp files
                                                                                        for (size_t i = 0; i < frames.size(); i++) {
                                                                                            std::string framePath = tmpDir + "/frame_" + std::to_string(i) + ".png";
                                                                                            remove(framePath.c_str());
                                                                                        }
                                                                                        rmdir(tmpDir.c_str());
                                                                                        frames.clear();
                                                                                    }
                                                                                };
                                                                            }

                                                                            static inline int pyDiv(int a,int b){return a/b-(a%b!=0&&((a^b)<0)?1:0);}
                                                                            static inline int pyMod(int a,int b){return a-b*pyDiv(a,b);}

                                                                            static inline float meanRGB(const uint8_t* p){return (p[0]+p[1]+p[2])/3.0f;}

                                                                            static inline float bt601(float r,float g,float b){return 0.299f*r+0.587f*g+0.114f*b;}

                                                                            static std::vector<int> argsortF(const std::vector<float>& v){
                                                                                std::vector<int> idx(v.size());std::iota(idx.begin(),idx.end(),0);
                                                                                std::sort(idx.begin(),idx.end(),[&](int a,int b){return v[a]<v[b];});
                                                                                return idx;
                                                                            }

                                                                            static cv::Mat applyTransforms(const cv::Mat& img,int rotSteps,bool flip){
                                                                                cv::Mat out=img.clone();
                                                                                if(flip) cv::flip(out,out,1);
                                                                                int s=rotSteps%4;
                                                                                if(s==1) cv::rotate(out,out,cv::ROTATE_90_COUNTERCLOCKWISE);
                                                                                else if(s==2) cv::rotate(out,out,cv::ROTATE_180);
                                                                                else if(s==3) cv::rotate(out,out,cv::ROTATE_90_CLOCKWISE);
                                                                                return out;
                                                                            }

                                                                            static cv::Mat applyUpscale(const cv::Mat& img,int tgt){
                                                                                int h=img.rows,w=img.cols,minD=std::min(h,w);
                                                                                int f=(int)((double)tgt/minD+0.9999);
                                                                                if(f>=2){cv::Mat o;cv::resize(img,o,cv::Size(w*f,h*f),0,0,cv::INTER_NEAREST);return o;}
                                                                                return img;
                                                                            }

                                                                            static uint32_t mortonSpread(uint32_t n){
                                                                                n=(n|(n<<16))&0x030000FF;
                                                                                n=(n|(n<< 8))&0x0300F00F;
                                                                                n=(n|(n<< 4))&0x030C30C3;
                                                                                n=(n|(n<< 2))&0x09249249;
                                                                                return n;
                                                                            }
                                                                            static uint32_t morton3(uint8_t r,uint8_t g,uint8_t b){
                                                                                return mortonSpread(r)|(mortonSpread(g)<<1)|(mortonSpread(b)<<2);
                                                                            }

                                                                            static std::vector<int> matchMorton(const std::vector<uint32_t>& sSorted,
                                                                                                                const std::vector<uint32_t>& tSorted){
                                                                                int sn=(int)sSorted.size(),tn=(int)tSorted.size();
                                                                                std::vector<int> m(tn);
                                                                                for(int i=0;i<tn;i++){
                                                                                    int p=(int)(std::lower_bound(sSorted.begin(),sSorted.end(),tSorted[i])-sSorted.begin());
                                                                                    m[i]=std::clamp(p,0,sn-1);
                                                                                }
                                                                                for(int i=1;i<tn;i++) if(m[i]<=m[i-1]) m[i]=m[i-1]+1;
                                                                                for(int i=0;i<tn;i++) m[i]=std::clamp(m[i],0,sn-1);
                                                                                return m;
                                                                                                                }

                                                                                                                static std::vector<float> quantize4(const std::vector<std::array<uint8_t,3>>& pixels){
                                                                                                                    float fac=255.f/3.f;
                                                                                                                    std::vector<float> r(pixels.size());
                                                                                                                    for(int i=0;i<(int)pixels.size();i++){
                                                                                                                        float g=bt601(pixels[i][0],pixels[i][1],pixels[i][2]);
                                                                                                                        r[i]=std::round(g/fac)*fac;
                                                                                                                    }
                                                                                                                    return r;
                                                                                                                }

                                                                                                                static std::vector<int32_t> assignPixels(const cv::Mat& baseRGB,const cv::Mat& tgtRGB,
                                                                                                                                                         const std::string& mode,const cv::Mat& maskFlat)
                                                                                                                {
                                                                                                                    int N=baseRGB.rows*baseRGB.cols;
                                                                                                                    std::vector<int32_t> asgn;
                                                                                                                    bool hasMask=(!maskFlat.empty()&&(int)maskFlat.total()==N);

                                                                                                                    if(mode=="disguise") {asgn.resize(N);std::iota(asgn.begin(),asgn.end(),0);}
                                                                                                                    else asgn.assign(N,-1);

                                                                                                                    const uint8_t* bp=baseRGB.data;
                                                                                                                    const uint8_t* tp=tgtRGB.data;

                                                                                                                    if((mode=="swap"||mode=="blend")&&hasMask){
                                                                                                                        const uint8_t* mp=maskFlat.data;
                                                                                                                        std::vector<int> validIdx,candidIdx;
                                                                                                                        for(int i=0;i<N;i++){if(mp[i]>0)validIdx.push_back(i);else candidIdx.push_back(i);}
                                                                                                                        if(validIdx.empty()||candidIdx.empty()) return asgn;
                                                                                                                        int cn=(int)candidIdx.size();
                                                                                                                        std::vector<uint32_t> sc(cn);
                                                                                                                        for(int i=0;i<cn;i++) sc[i]=morton3(bp[candidIdx[i]*3],bp[candidIdx[i]*3+1],bp[candidIdx[i]*3+2]);
                                                                                                                        std::vector<int> ssIdx(cn);std::iota(ssIdx.begin(),ssIdx.end(),0);
                                                                                                                        std::sort(ssIdx.begin(),ssIdx.end(),[&](int a,int b){return sc[a]<sc[b];});
                                                                                                                        std::vector<uint32_t> scSorted(cn);
                                                                                                                        for(int i=0;i<cn;i++) scSorted[i]=sc[ssIdx[i]];
                                                                                                                        int vn=(int)validIdx.size();
                                                                                                                        std::vector<uint32_t> tc(vn);
                                                                                                                        for(int i=0;i<vn;i++) tc[i]=morton3(tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]);
                                                                                                                        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
                                                                                                                        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tc[a]<tc[b];});
                                                                                                                        std::vector<uint32_t> tcSorted(vn);
                                                                                                                        for(int i=0;i<vn;i++) tcSorted[i]=tc[tsIdx[i]];
                                                                                                                        auto mi=matchMorton(scSorted,tcSorted);
                                                                                                                        for(int i=0;i<vn;i++){
                                                                                                                            int sg=candidIdx[ssIdx[mi[i]]];
                                                                                                                            int tg=validIdx[tsIdx[i]];
                                                                                                                            asgn[sg]=tg; asgn[tg]=sg;
                                                                                                                        }
                                                                                                                        return asgn;
                                                                                                                    }

                                                                                                                    if(mode=="navigate"&&hasMask){
                                                                                                                        const uint8_t* mp=maskFlat.data;
                                                                                                                        std::vector<int> validIdx;
                                                                                                                        for(int i=0;i<N;i++) if(mp[i]>0) validIdx.push_back(i);
                                                                                                                        if(validIdx.empty()) return asgn;
                                                                                                                        std::vector<uint32_t> sc(N);
                                                                                                                        for(int i=0;i<N;i++) sc[i]=morton3(bp[i*3],bp[i*3+1],bp[i*3+2]);
                                                                                                                        std::vector<int> ssIdx(N);std::iota(ssIdx.begin(),ssIdx.end(),0);
                                                                                                                        std::sort(ssIdx.begin(),ssIdx.end(),[&](int a,int b){return sc[a]<sc[b];});
                                                                                                                        std::vector<uint32_t> scSorted(N);
                                                                                                                        for(int i=0;i<N;i++) scSorted[i]=sc[ssIdx[i]];
                                                                                                                        int vn=(int)validIdx.size();
                                                                                                                        std::vector<uint32_t> tc(vn);
                                                                                                                        for(int i=0;i<vn;i++) tc[i]=morton3(tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]);
                                                                                                                        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
                                                                                                                        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tc[a]<tc[b];});
                                                                                                                        std::vector<uint32_t> tcSorted(vn);
                                                                                                                        for(int i=0;i<vn;i++) tcSorted[i]=tc[tsIdx[i]];
                                                                                                                        auto mi=matchMorton(scSorted,tcSorted);
                                                                                                                        for(int i=0;i<vn;i++) asgn[ssIdx[mi[i]]]=validIdx[tsIdx[i]];
                                                                                                                        return asgn;
                                                                                                                    }

                                                                                                                    if(mode=="pattern"&&hasMask){
                                                                                                                        const uint8_t* mp=maskFlat.data;
                                                                                                                        std::vector<int> validIdx;
                                                                                                                        for(int i=0;i<N;i++) if(mp[i]>0) validIdx.push_back(i);
                                                                                                                        if(validIdx.empty()) return asgn;
                                                                                                                        int vn=(int)validIdx.size();
                                                                                                                        std::vector<float> sg(N);
                                                                                                                        for(int i=0;i<N;i++) sg[i]=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
                                                                                                                        auto ssIdx=argsortF(sg);
                                                                                                                        std::vector<std::array<uint8_t,3>> tVpx(vn);
                                                                                                                        for(int i=0;i<vn;i++){tVpx[i]={tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]};}
                                                                                                                        auto tq=quantize4(tVpx);
                                                                                                                        static std::mt19937 noiseRng(12345);
                                                                                                                        std::uniform_real_distribution<float> nd(0.f,.5f);
                                                                                                                        for(float& v:tq) v+=nd(noiseRng);
                                                                                                                        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
                                                                                                                        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tq[a]<tq[b];});
                                                                                                                        for(int i=0;i<vn;i++){
                                                                                                                            float val=(vn>1)?(float)i*(N-1.f)/(vn-1.f):0.f;
                                                                                                                            int slin=(int)val;
                                                                                                                            asgn[ssIdx[slin]]=validIdx[tsIdx[i]];
                                                                                                                        }
                                                                                                                        return asgn;
                                                                                                                    }

                                                                                                                    if(mode=="disguise"&&hasMask){
                                                                                                                        const uint8_t* mp=maskFlat.data;
                                                                                                                        std::vector<int> vi; for(int i=0;i<N;i++) if(mp[i]>0) vi.push_back(i);
                                                                                                                        if(vi.empty()) return asgn;
                                                                                                                        int vn=(int)vi.size();
                                                                                                                        std::vector<float> sg(vn),tg(vn);
                                                                                                                        for(int i=0;i<vn;i++){sg[i]=(bp[vi[i]*3]+bp[vi[i]*3+1]+bp[vi[i]*3+2])/3.f;
                                                                                                                            tg[i]=(tp[vi[i]*3]+tp[vi[i]*3+1]+tp[vi[i]*3+2])/3.f;}
                                                                                                                            auto ss=argsortF(sg),ts=argsortF(tg);
                                                                                                                            for(int i=0;i<vn;i++) asgn[vi[ss[i]]]=vi[ts[i]];
                                                                                                                            return asgn;
                                                                                                                    }

                                                                                                                    if(mode=="fusion"&&hasMask){
                                                                                                                        const uint8_t* mp=maskFlat.data;
                                                                                                                        std::vector<int> vi; for(int i=0;i<N;i++) if(mp[i]>0) vi.push_back(i);
                                                                                                                        if(vi.empty()) return asgn;
                                                                                                                        int vn=(int)vi.size();
                                                                                                                        std::vector<float> sg(vn),tg(vn);
                                                                                                                        for(int i=0;i<vn;i++){sg[i]=(bp[vi[i]*3]+bp[vi[i]*3+1]+bp[vi[i]*3+2])/3.f;
                                                                                                                            tg[i]=(tp[vi[i]*3]+tp[vi[i]*3+1]+tp[vi[i]*3+2])/3.f;}
                                                                                                                            auto ss=argsortF(sg),ts=argsortF(tg);
                                                                                                                            for(int i=0;i<vn;i++) asgn[vi[ss[i]]]=vi[ts[i]];
                                                                                                                            return asgn;
                                                                                                                    }

                                                                                                                    if(mode=="shuffle"){
                                                                                                                        static std::mt19937 rng(std::random_device{}());
                                                                                                                        std::vector<int> sBlk,sWht,tBlk,tWht;
                                                                                                                        for(int i=0;i<N;i++){
                                                                                                                            float sg=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
                                                                                                                            float tg=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;
                                                                                                                            if(sg>127.f) sWht.push_back(i); else sBlk.push_back(i);
                                                                                                                            if(tg>127.f) tWht.push_back(i); else tBlk.push_back(i);
                                                                                                                        }
                                                                                                                        std::shuffle(sBlk.begin(),sBlk.end(),rng);std::shuffle(sWht.begin(),sWht.end(),rng);
                                                                                                                        std::shuffle(tBlk.begin(),tBlk.end(),rng);std::shuffle(tWht.begin(),tWht.end(),rng);
                                                                                                                        int mb=(int)std::min(sBlk.size(),tBlk.size()),mw=(int)std::min(sWht.size(),tWht.size());
                                                                                                                        for(int i=0;i<mb;i++) asgn[sBlk[i]]=tBlk[i];
                                                                                                                        for(int i=0;i<mw;i++) asgn[sWht[i]]=tWht[i];
                                                                                                                        std::vector<int> sRem,tRem;
                                                                                                                        for(int i=mb;i<(int)sBlk.size();i++) sRem.push_back(sBlk[i]);
                                                                                                                        for(int i=mw;i<(int)sWht.size();i++) sRem.push_back(sWht[i]);
                                                                                                                        for(int i=mb;i<(int)tBlk.size();i++) tRem.push_back(tBlk[i]);
                                                                                                                        for(int i=mw;i<(int)tWht.size();i++) tRem.push_back(tWht[i]);
                                                                                                                        if(!sRem.empty()&&!tRem.empty()){
                                                                                                                            std::shuffle(sRem.begin(),sRem.end(),rng);
                                                                                                                            std::shuffle(tRem.begin(),tRem.end(),rng);
                                                                                                                            int mn=(int)std::min(sRem.size(),tRem.size());
                                                                                                                            for(int i=0;i<mn;i++) asgn[sRem[i]]=tRem[i];
                                                                                                                        }
                                                                                                                        return asgn;
                                                                                                                    }

                                                                                                                    {
                                                                                                                        std::vector<float> sg(N),tg(N);
                                                                                                                        for(int i=0;i<N;i++){sg[i]=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
                                                                                                                            tg[i]=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;}
                                                                                                                            auto ss=argsortF(sg),ts=argsortF(tg);
                                                                                                                            for(int i=0;i<N;i++) asgn[ss[i]]=ts[i];
                                                                                                                    }
                                                                                                                    return asgn;
                                                                                                                }

                                                                                                                static std::vector<int32_t> assignDrawerPixels(const cv::Mat& baseRGB,const cv::Mat& tgtRGB){
                                                                                                                    int N=baseRGB.rows*baseRGB.cols;
                                                                                                                    std::vector<int32_t> asgn(N);std::iota(asgn.begin(),asgn.end(),0);
                                                                                                                    const uint8_t* bp=baseRGB.data,*tp=tgtRGB.data;
                                                                                                                    std::vector<int> nw;
                                                                                                                    for(int i=0;i<N;i++) if(bp[i*3]<245||bp[i*3+1]<245||bp[i*3+2]<245) nw.push_back(i);
                                                                                                                    if(nw.empty()) return asgn;
                                                                                                                    std::vector<float> bg(nw.size());
                                                                                                                    for(int i=0;i<(int)nw.size();i++) bg[i]=(bp[nw[i]*3]+bp[nw[i]*3+1]+bp[nw[i]*3+2])/3.f;
                                                                                                                    auto bsort=argsortF(bg);
                                                                                                                    std::vector<float> tg(N);
                                                                                                                    for(int i=0;i<N;i++) tg[i]=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;
                                                                                                                    auto tsort=argsortF(tg);
                                                                                                                    int nb=(int)nw.size();
                                                                                                                    if(nb<=N){
                                                                                                                        for(int i=0;i<nb;i++) asgn[nw[bsort[i]]]=tsort[std::min(i,N-1)];
                                                                                                                    } else {
                                                                                                                        std::set<int> used;
                                                                                                                        for(int i=0;i<nb;i++){
                                                                                                                            int st=std::min(i,N-1),idx=st;
                                                                                                                            while(idx<N&&used.count(tsort[idx])) idx++;
                                                                                                                            if(idx>=N) idx=N-1;
                                                                                                                            used.insert(tsort[idx]);
                                                                                                                            asgn[nw[bsort[i]]]=tsort[idx];
                                                                                                                        }
                                                                                                                    }
                                                                                                                    return asgn;
                                                                                                                }

                                                                                                                class Missform {
                                                                                                                public:
                                                                                                                    int H,W,minP=0;
                                                                                                                    std::vector<std::array<int,2>> bPos,tPos;
                                                                                                                    std::vector<std::array<uint8_t,3>> bCol;

                                                                                                                    Missform(const cv::Mat& bRGB,const cv::Mat& tRGB,float thr=127.f){
                                                                                                                        H=bRGB.rows;W=bRGB.cols;
                                                                                                                        auto mask=[&](const cv::Mat& img,std::vector<bool>& m){
                                                                                                                            m.resize(img.rows*img.cols);
                                                                                                                            for(int i=0;i<(int)m.size();i++) m[i]=(img.data[i*3]+img.data[i*3+1]+img.data[i*3+2])/3.f>thr;
                                                                                                                        };
                                                                                                                            std::vector<bool> bm,tm; mask(bRGB,bm); mask(tRGB,tm);
                                                                                                                            for(int y=0;y<H;y++) for(int x=0;x<W;x++){
                                                                                                                                if(bm[y*W+x]) bPos.push_back({y,x});
                                                                                                                                if(tm[y*W+x]) tPos.push_back({y,x});
                                                                                                                            }
                                                                                                                            minP=(int)std::min(bPos.size(),tPos.size());
                                                                                                                            if(minP==0) throw std::runtime_error("No valid pixels found for morphing");
                                                                                                                            bPos.resize(minP);tPos.resize(minP);bCol.resize(minP);
                                                                                                                        for(int i=0;i<minP;i++){
                                                                                                                            const uint8_t* p=bRGB.data+(bPos[i][0]*W+bPos[i][1])*3;
                                                                                                                            bCol[i]={p[0],p[1],p[2]};
                                                                                                                        }
                                                                                                                    }
                                                                                                                    cv::Mat frame(float progress) const {
                                                                                                                        cv::Mat f(H,W,CV_8UC3,cv::Scalar(0,0,0));
                                                                                                                        if(minP==0) return f;
                                                                                                                        float t=progress*progress*(3.f-2.f*progress);
                                                                                                                        for(int i=0;i<minP;i++){
                                                                                                                            int cy=(int)(bPos[i][0]+(tPos[i][0]-bPos[i][0])*t);
                                                                                                                            int cx=(int)(bPos[i][1]+(tPos[i][1]-bPos[i][1])*t);
                                                                                                                            if(cy>=0&&cy<H&&cx>=0&&cx<W){
                                                                                                                                uint8_t* p=f.data+(cy*W+cx)*3;
                                                                                                                                p[0]=bCol[i][0];p[1]=bCol[i][1];p[2]=bCol[i][2];
                                                                                                                            }
                                                                                                                        }
                                                                                                                        return f;
                                                                                                                    }
                                                                                                                };

                                                                                                                struct ProcessConfig {
                                                                                                                    std::string basePath,tgtPath;
                                                                                                                    std::string mode;
                                                                                                                    std::string algo;
                                                                                                                    int   rotBase=0;  bool flipBase=false;
                                                                                                                    int   rotTgt=0;   bool flipTgt=false;
                                                                                                                    cv::Mat mask;
                                                                                                                    int   resolution=512;
                                                                                                                    std::string soundOpt="mute";
                                                                                                                    int   audioQuality=30;
                                                                                                                    cv::Mat baseImageArray;
                                                                                                                    int   fps=30;
                                                                                                                    bool* running=nullptr;
                                                                                                                };

                                                                                                                static void processCore(const ProcessConfig& cfg,
                                                                                                                                        std::function<void(int)>           onProgress,
                                                                                                                                        std::function<void(const QImage&)> onFrame,
                                                                                                                                        std::function<void(const std::string&)> onFinish,
                                                                                                                                        std::function<void(const std::string&)> onError)
                                                                                                                {
                                                                                                                    try {
                                                                                                                        static const std::vector<std::string> maskModes={"pattern","disguise","navigate","swap","blend"};
                                                                                                                        bool needsMask=std::find(maskModes.begin(),maskModes.end(),cfg.algo)!=maskModes.end();
                                                                                                                        if(needsMask&&cfg.mask.empty())
                                                                                                                            throw std::runtime_error(cfg.algo+" mode requires analyzing shapes first.");
                                                                                                                        if(cfg.algo=="drawer"&&cfg.baseImageArray.empty())
                                                                                                                            throw std::runtime_error("Drawer mode requires base drawing data.");

                                                                                                                        const std::string outDir="results";
                                                                                                                        mkdir(outDir.c_str(),0755);

                                                                                                                        cv::Mat baseImg,tgtImg;
                                                                                                                        if(cfg.algo=="drawer") baseImg=cfg.baseImageArray.clone();
                                                                                                                        else baseImg=cv::imread(cfg.basePath);
                                                                                                                        tgtImg=cv::imread(cfg.tgtPath);
                                                                                                                        if(baseImg.empty()||tgtImg.empty()) throw std::runtime_error("Could not load images");

                                                                                                                        baseImg=applyUpscale(baseImg,cfg.resolution);
                                                                                                                        tgtImg =applyUpscale(tgtImg, cfg.resolution);
                                                                                                                        int limitRes=std::min({baseImg.rows,baseImg.cols,tgtImg.rows,tgtImg.cols});
                                                                                                                        int procRes=std::min(cfg.resolution,limitRes);

                                                                                                                        if(cfg.algo!="drawer") baseImg=applyTransforms(baseImg,cfg.rotBase,cfg.flipBase);
                                                                                                                        tgtImg=applyTransforms(tgtImg,cfg.rotTgt,cfg.flipTgt);
                                                                                                                        cv::resize(baseImg,baseImg,cv::Size(procRes,procRes));
                                                                                                                        cv::resize(tgtImg, tgtImg, cv::Size(procRes,procRes));
                                                                                                                        cv::cvtColor(baseImg,baseImg,cv::COLOR_BGR2RGB);
                                                                                                                        cv::cvtColor(tgtImg, tgtImg, cv::COLOR_BGR2RGB);

                                                                                                                        int W=procRes,H=procRes,N=W*H;
                                                                                                                        int totalFrames=cfg.fps*10;

                                                                                                                        time_t now2=time(nullptr);
                                                                                                                        char ts[32];strftime(ts,sizeof(ts),"%Y%m%d_%H%M%S",localtime(&now2));

                                                                                                                        if(cfg.algo=="missform"){
                                                                                                                            Missform miss(baseImg,tgtImg,127.f);
                                                                                                                            std::string outPath,silPath;
                                                                                                                            cv::VideoWriter vw; GIF::Encoder gifEnc;
                                                                                                                            std::vector<cv::Mat> vframes;
                                                                                                                            if(cfg.mode=="export_video"){
                                                                                                                                outPath=outDir+"/video_"+ts+".mp4";
                                                                                                                                int fcc=cv::VideoWriter::fourcc('m','p','4','v');
                                                                                                                                if(cfg.soundOpt!="mute"){silPath=outDir+"/video_"+ts+"_silent.mp4";vw.open(silPath,fcc,30.0,cv::Size(W,H));}
                                                                                                                                else vw.open(outPath,fcc,30.0,cv::Size(W,H));
                                                                                                                            } else if(cfg.mode=="export_gif"){
                                                                                                                                outPath=outDir+"/animation_"+ts+".gif";gifEnc.open(outPath,W,H,1000/cfg.fps);
                                                                                                                            }
                                                                                                                            for(int f=0;f<totalFrames;f++){
                                                                                                                                if(cfg.running&&!*cfg.running) break;
                                                                                                                                float progress=(float)f/(totalFrames-1);
                                                                                                                                onProgress((int)(progress*100));
                                                                                                                                float t=progress*progress*(3.f-2.f*progress);
                                                                                                                                cv::Mat frm=miss.frame(t);
                                                                                                                                if(cfg.mode=="preview"){
                                                                                                                                    QImage qi(frm.data,W,H,W*3,QImage::Format_RGB888);onFrame(qi.copy());
                                                                                                                                    struct timespec sl={0,16666666};nanosleep(&sl,nullptr);
                                                                                                                                } else if(cfg.mode=="export_video"){
                                                                                                                                    if(cfg.soundOpt!="mute") vframes.push_back(frm.clone());
                                                                                                                                    cv::Mat bgr;cv::cvtColor(frm,bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                                                                                                                                } else if(cfg.mode=="export_gif") gifEnc.writeFrame(frm);
                                                                                                                            }
                                                                                                                            if(cfg.mode=="export_video"){
                                                                                                                                vw.release();
                                                                                                                                if(cfg.soundOpt!="mute"){
                                                                                                                                    std::string tgtA=(cfg.soundOpt=="target-sound")?cfg.tgtPath:"";
                                                                                                                                    std::string fin=addAudioToVideo(silPath,vframes,(double)cfg.fps,outPath,cfg.soundOpt,tgtA,cfg.audioQuality);
                                                                                                                                    remove(silPath.c_str());onFinish("Saved to "+fin);
                                                                                                                                } else onFinish("Saved to "+outPath);
                                                                                                                            } else if(cfg.mode=="export_image"){
                                                                                                                                std::string ip=outDir+"/image_"+ts+".png";
                                                                                                                                cv::Mat fin=miss.frame(1.f);
                                                                                                                                cv::Mat bgr;cv::cvtColor(fin,bgr,cv::COLOR_RGB2BGR);
                                                                                                                                cv::imwrite(ip,bgr);onFinish("Saved to "+ip);
                                                                                                                            } else if(cfg.mode=="export_gif"){gifEnc.close();onFinish("Saved to "+outPath);}
                                                                                                                            else onFinish("Preview finished");
                                                                                                                            return;
                                                                                                                        }

                                                                                                                        cv::Mat procMask;
                                                                                                                        if(!cfg.mask.empty()){
                                                                                                                            cv::Mat m8; cfg.mask.convertTo(m8,CV_8U,255.0);
                                                                                                                            cv::resize(m8,m8,cv::Size(W,H),0,0,cv::INTER_NEAREST);
                                                                                                                            procMask=m8.reshape(1,N);
                                                                                                                        }

                                                                                                                        std::vector<int32_t> asgn;
                                                                                                                        if(cfg.algo=="drawer") asgn=assignDrawerPixels(baseImg,tgtImg);
                                                                                                                        else asgn=assignPixels(baseImg,tgtImg,cfg.algo,procMask);

                                                                                                                        std::vector<int> validIdx;
                                                                                                                        bool allValid=(cfg.algo=="disguise"||cfg.algo=="shuffle"||cfg.algo=="merge"||cfg.algo=="drawer");
                                                                                                                        if(allValid){validIdx.resize(N);std::iota(validIdx.begin(),validIdx.end(),0);}
                                                                                                                        else for(int i=0;i<N;i++) if(asgn[i]>=0) validIdx.push_back(i);
                                                                                                                        int VN=(int)validIdx.size();

                                                                                                                        std::vector<int> sX(VN),sY(VN),eX(VN),eY(VN);
                                                                                                                        std::vector<std::array<float,3>> srcC(VN),tgtCA(VN);
                                                                                                                        const uint8_t* bp2=baseImg.data,*tp2=tgtImg.data;
                                                                                                                        for(int i=0;i<VN;i++){
                                                                                                                            int vi=validIdx[i];
                                                                                                                            sY[i]=vi/W;sX[i]=vi%W;
                                                                                                                            int d=asgn[vi];
                                                                                                                            eY[i]=pyDiv(d,W);eX[i]=pyMod(d,W);
                                                                                                                            srcC[i]={bp2[vi*3+0]*1.f,bp2[vi*3+1]*1.f,bp2[vi*3+2]*1.f};
                                                                                                                            if(d>=0) tgtCA[i]={tp2[d*3+0]*1.f,tp2[d*3+1]*1.f,tp2[d*3+2]*1.f};
                                                                                                                        }

                                                                                                                        std::vector<std::array<float,3>> shuffSrc;
                                                                                                                        if(cfg.algo=="fusion"){
                                                                                                                            shuffSrc=srcC;
                                                                                                                            std::mt19937 rng42(42);
                                                                                                                            std::shuffle(shuffSrc.begin(),shuffSrc.end(),rng42);
                                                                                                                        }

                                                                                                                        std::vector<float> tgtGrayF,srcLuma,posX,posY;
                                                                                                                        cv::Mat gradX,gradY;
                                                                                                                        if(cfg.algo=="blend"){
                                                                                                                            cv::Mat tg; cv::cvtColor(tgtImg,tg,cv::COLOR_RGB2GRAY);
                                                                                                                            tg.convertTo(tg,CV_32F,1./255.);
                                                                                                                            cv::Sobel(tg,gradX,CV_32F,1,0,3);
                                                                                                                            cv::Sobel(tg,gradY,CV_32F,0,1,3);
                                                                                                                            tgtGrayF.resize(N);memcpy(tgtGrayF.data(),tg.data,N*sizeof(float));
                                                                                                                            srcLuma.resize(VN);
                                                                                                                            for(int i=0;i<VN;i++) srcLuma[i]=(0.299f*srcC[i][0]+0.587f*srcC[i][1]+0.114f*srcC[i][2])/255.f;
                                                                                                                            posX.resize(VN);posY.resize(VN);
                                                                                                                            for(int i=0;i<VN;i++){posX[i]=(float)sX[i];posY[i]=(float)sY[i];}
                                                                                                                        }

                                                                                                                        std::vector<bool> unmasked(N,false);
                                                                                                                        if(cfg.algo=="fusion"&&!procMask.empty())
                                                                                                                            for(int i=0;i<N;i++) unmasked[i]=(procMask.data[i]==0);

                                                                                                                            std::string outPath,silPath;
                                                                                                                        cv::VideoWriter vw; GIF::Encoder gifEnc;
                                                                                                                        std::vector<cv::Mat> vframes;
                                                                                                                        if(cfg.mode=="export_video"){
                                                                                                                            outPath=outDir+"/video_"+ts+".mp4";
                                                                                                                            int fcc=cv::VideoWriter::fourcc('m','p','4','v');
                                                                                                                            if(cfg.soundOpt!="mute"){silPath=outDir+"/video_"+ts+"_silent.mp4";vw.open(silPath,fcc,30.0,cv::Size(W,H));}
                                                                                                                            else vw.open(outPath,fcc,(double)cfg.fps,cv::Size(W,H));
                                                                                                                        } else if(cfg.mode=="export_gif"){
                                                                                                                            outPath=outDir+"/animation_"+ts+".gif";gifEnc.open(outPath,W,H,1000/cfg.fps);
                                                                                                                        }

                                                                                                                        cv::Mat lastFrame;
                                                                                                                        for(int f=0;f<totalFrames;f++){
                                                                                                                            if(cfg.running&&!*cfg.running) break;
                                                                                                                            float progress=(float)f/std::max(1,totalFrames-1);
                                                                                                                            onProgress((int)(progress*100));
                                                                                                                            float t=progress*progress*(3.f-2.f*progress);

                                                                                                                            std::vector<int> cx(VN),cy(VN);
                                                                                                                            if(cfg.algo=="blend"){
                                                                                                                                float kD=0.05f+progress*0.5f,kH=0.05f*(1.f-progress),kG=6.f*(1.f-progress*0.2f);
                                                                                                                                for(int i=0;i<VN;i++){
                                                                                                                                    int cxi=std::clamp((int)posX[i],0,W-1);
                                                                                                                                    int cyi=std::clamp((int)posY[i],0,H-1);
                                                                                                                                    float tl=tgtGrayF[cyi*W+cxi];
                                                                                                                                    float diff=srcLuma[i]-tl;
                                                                                                                                    float gx=((float*)gradX.data)[cyi*W+cxi];
                                                                                                                                    float gy=((float*)gradY.data)[cyi*W+cxi];
                                                                                                                                    posX[i]+=gx*diff*kG+(sX[i]-posX[i])*kH+(eX[i]-posX[i])*kD;
                                                                                                                                    posY[i]+=gy*diff*kG+(sY[i]-posY[i])*kH+(eY[i]-posY[i])*kD;
                                                                                                                                    cx[i]=std::clamp((int)posX[i],0,W-1);
                                                                                                                                    cy[i]=std::clamp((int)posY[i],0,H-1);
                                                                                                                                }
                                                                                                                            } else {
                                                                                                                                for(int i=0;i<VN;i++){
                                                                                                                                    cx[i]=std::clamp((int)(sX[i]+(eX[i]-sX[i])*t),0,W-1);
                                                                                                                                    cy[i]=std::clamp((int)(sY[i]+(eY[i]-sY[i])*t),0,H-1);
                                                                                                                                }
                                                                                                                            }

                                                                                                                            cv::Mat frm;
                                                                                                                            if(cfg.algo=="disguise"||cfg.algo=="fusion") frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));
                                                                                                                            else if(cfg.algo=="drawer")                  frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(255,255,255));
                                                                                                                            else if(cfg.algo=="navigate"||cfg.algo=="swap"||cfg.algo=="blend") frm=baseImg.clone();
                                                                                                                            else frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));

                                                                                                                            if(cfg.algo=="fusion"&&!procMask.empty())
                                                                                                                                for(int i=0;i<N;i++) if(unmasked[i]){
                                                                                                                                    int y=i/W,x=i%W;
                                                                                                                                    uint8_t* p=frm.data+(y*W+x)*3;
                                                                                                                                    p[0]=bp2[i*3];p[1]=bp2[i*3+1];p[2]=bp2[i*3+2];
                                                                                                                                }

                                                                                                                                for(int i=0;i<VN;i++){
                                                                                                                                    std::array<float,3> c;
                                                                                                                                    if(cfg.algo=="fusion"){
                                                                                                                                        const auto& s2=(cfg.mask.empty()?srcC:shuffSrc)[i];
                                                                                                                                        const auto& td=tgtCA[i];
                                                                                                                                        c={s2[0]*(1-t)+td[0]*t,s2[1]*(1-t)+td[1]*t,s2[2]*(1-t)+td[2]*t};
                                                                                                                                    } else c=srcC[i];
                                                                                                                                    uint8_t* p=frm.data+(cy[i]*W+cx[i])*3;
                                                                                                                                    p[0]=(uint8_t)std::clamp(c[0],0.f,255.f);
                                                                                                                                    p[1]=(uint8_t)std::clamp(c[1],0.f,255.f);
                                                                                                                                    p[2]=(uint8_t)std::clamp(c[2],0.f,255.f);
                                                                                                                                }
                                                                                                                                lastFrame=frm;

                                                                                                                                if(cfg.mode=="preview"){
                                                                                                                                    QImage qi(frm.data,W,H,W*3,QImage::Format_RGB888);onFrame(qi.copy());
                                                                                                                                    struct timespec sl={0,16666666};nanosleep(&sl,nullptr);
                                                                                                                                } else if(cfg.mode=="export_video"){
                                                                                                                                    if(cfg.soundOpt!="mute") vframes.push_back(frm.clone());
                                                                                                                                    cv::Mat bgr;cv::cvtColor(frm,bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                                                                                                                                } else if(cfg.mode=="export_gif") gifEnc.writeFrame(frm);
                                                                                                                        }

                                                                                                                        if(cfg.mode=="export_video"){
                                                                                                                            vw.release();
                                                                                                                            if(cfg.soundOpt!="mute"){
                                                                                                                                std::string tgtA=(cfg.soundOpt=="target-sound")?cfg.tgtPath:"";
                                                                                                                                std::string fin=addAudioToVideo(silPath,vframes,(double)cfg.fps,outPath,cfg.soundOpt,tgtA,cfg.audioQuality);
                                                                                                                                remove(silPath.c_str());onFinish("Saved to "+fin);
                                                                                                                            } else onFinish("Saved to "+outPath);
                                                                                                                        } else if(cfg.mode=="export_image"){
                                                                                                                            std::string ip=outDir+"/image_"+ts+".png";
                                                                                                                            cv::Mat fin;
                                                                                                                            if(cfg.algo=="fusion"&&!procMask.empty()){
                                                                                                                                fin=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));
                                                                                                                                for(int i=0;i<N;i++) if(unmasked[i]){
                                                                                                                                    int y=i/W,x=i%W;uint8_t* p=fin.data+(y*W+x)*3;
                                                                                                                                    p[0]=bp2[i*3];p[1]=bp2[i*3+1];p[2]=bp2[i*3+2];
                                                                                                                                }
                                                                                                                            } else fin=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));
                                                                                                                            for(int i=0;i<VN;i++){
                                                                                                                                std::array<float,3> c=srcC[i];
                                                                                                                                if(cfg.algo=="fusion"){
                                                                                                                                    const auto& s2=(cfg.mask.empty()?srcC:shuffSrc)[i];
                                                                                                                                    c={tgtCA[i][0],tgtCA[i][1],tgtCA[i][2]};(void)s2;
                                                                                                                                }
                                                                                                                                int y=eY[i],x=eX[i];
                                                                                                                                if(y>=0&&y<H&&x>=0&&x<W){
                                                                                                                                    uint8_t* p=fin.data+(y*W+x)*3;
                                                                                                                                    p[0]=(uint8_t)std::clamp(c[0],0.f,255.f);
                                                                                                                                    p[1]=(uint8_t)std::clamp(c[1],0.f,255.f);
                                                                                                                                    p[2]=(uint8_t)std::clamp(c[2],0.f,255.f);
                                                                                                                                }
                                                                                                                            }
                                                                                                                            cv::Mat bgr;cv::cvtColor(fin,bgr,cv::COLOR_RGB2BGR);
                                                                                                                            cv::imwrite(ip,bgr);onFinish("Saved to "+ip);
                                                                                                                        } else if(cfg.mode=="export_gif"){gifEnc.close();onFinish("Saved to "+outPath);}
                                                                                                                        else onFinish("Preview finished");

                                                                                                                    } catch(const std::exception& ex){ onError(ex.what()); }
                                                                                                                }

                                                                                                                static bool isVideoFile(const std::string& p){
                                                                                                                    static const char* E[]={"mp4","avi","mov","mkv","flv","wmv"};
                                                                                                                    size_t d=p.rfind('.');if(d==std::string::npos)return false;
                                                                                                                    std::string e=p.substr(d+1);std::transform(e.begin(),e.end(),e.begin(),::tolower);
                                                                                                                    for(auto x:E) if(e==x) return true; return false;
                                                                                                                }

                                                                                                                static bool validateMediaFile(const std::string& path){
                                                                                                                    if(access(path.c_str(),F_OK)!=0){fprintf(stderr,"Error: File not found: %s\n",path.c_str());return false;}
                                                                                                                    static const char* E[]={".png",".jpg",".jpeg",".webp",".mp4",".avi",".mov",".mkv",".flv",".wmv"};
                                                                                                                    size_t d=path.rfind('.');if(d==std::string::npos){fprintf(stderr,"Error: No file extension.\n");return false;}
                                                                                                                    std::string e=path.substr(d);std::transform(e.begin(),e.end(),e.begin(),::tolower);
                                                                                                                    for(auto x:E) if(e==x) return true;
                                                                                                                    fprintf(stderr,"Error: Invalid file format. Supported: png jpg jpeg webp mp4 avi mov mkv flv wmv\n");
                                                                                                                    return false;
                                                                                                                }

                                                                                                                static void printBanner(){}

                                                                                                                static void printProgressBar(int iter,int total,const char* prefix="Progress:",int length=50){
                                                                                                                    float pct=total>0?100.f*(float)iter/(float)total:100.f;
                                                                                                                    int filled=total>0?(int)((float)length*iter/total):length;
                                                                                                                    std::string bstr(filled,'#');
                                                                                                                    bstr+=std::string(length-filled,'-');
                                                                                                                    if(iter==total) printf("\n");
                                                                                                                }

                                                                                                                static std::string getInputLine(const char* prompt,std::function<bool(const std::string&)> validate=nullptr){
                                                                                                                    while(true){
                                                                                                                        printf("%s",prompt);fflush(stdout);
                                                                                                                        std::string line;
                                                                                                                        if(!std::getline(std::cin,line)){printf("\nOperation cancelled.\n");exit(0);}
                                                                                                                        size_t s=line.find_first_not_of(" \t\r\n"),e=line.find_last_not_of(" \t\r\n");
                                                                                                                        if(s==std::string::npos) line=""; else line=line.substr(s,e-s+1);
                                                                                                                        if(!validate||validate(line)) return line;
                                                                                                                        printf("Invalid input. Please try again.\n");
                                                                                                                    }
                                                                                                                }

                                                                                                                static std::string selectAlgorithm(){
                                                                                                                    printf("\nSelect Algorithm:\n");
                                                                                                                    printf("1. Shuffle   - Randomly swap pixels between images\n");
                                                                                                                    printf("2. Merge     - Blend images with grayscale sorting\n");
                                                                                                                    printf("3. Missform  - Morph between binary pixel positions\n");
                                                                                                                    printf("4. Fusion    - Create animation with pixel sorting\n");
                                                                                                                    static const char* A[]={"shuffle","merge","missform","fusion"};
                                                                                                                    while(true){
                                                                                                                        printf("\nEnter your choice (1-4): ");fflush(stdout);
                                                                                                                        std::string c;std::getline(std::cin,c);
                                                                                                                        if(c>="1"&&c<="4") return A[c[0]-'1'];
                                                                                                                        printf("Invalid choice. Please enter 1, 2, 3, or 4.\n");
                                                                                                                    }
                                                                                                                }

                                                                                                                static int selectResolution(){
                                                                                                                    printf("\nSelect Resolution:\n1. 128x128\n2. 256x256\n3. 512x512\n4. 768x768\n5. 1024x1024\n6. 2048x2048\n");
                                                                                                                    static int R[]={128,256,512,768,1024,2048};
                                                                                                                    while(true){
                                                                                                                        printf("\nEnter your choice (1-6): ");fflush(stdout);
                                                                                                                        std::string c;std::getline(std::cin,c);
                                                                                                                        if(c>="1"&&c<="6") return R[c[0]-'1'];
                                                                                                                        printf("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.\n");
                                                                                                                    }
                                                                                                                }

                                                                                                                static std::pair<std::string,int> selectTargetSoundQuality(){
                                                                                                                    printf("\nSelect Target Sound Quality (1-10, where 10=100%% original quality, 3=30%% default):\n");
                                                                                                                    printf("1. 10%% (lowest quality)\n2. 20%%\n3. 30%% (default)\n4. 40%%\n5. 50%%\n");
                                                                                                                    printf("6. 60%%\n7. 70%%\n8. 80%%\n9. 90%%\n10. 100%% (original quality)\n");
                                                                                                                    while(true){
                                                                                                                        printf("Enter your choice (1-10, default 3): ");fflush(stdout);
                                                                                                                        std::string c;std::getline(std::cin,c);
                                                                                                                        if(c.empty()) return {"target-sound",30};
                                                                                                                        try{int q=std::stoi(c);if(q>=1&&q<=10)return{"target-sound",q*10};
                                                                                                                        printf("Please enter a number between 1 and 10.\n");
                                                                                                                        }catch(...){printf("Please enter a valid number.\n");}
                                                                                                                    }
                                                                                                                }

                                                                                                                static std::pair<std::string,int> selectSoundOption(bool tgtIsVid){
                                                                                                                    printf("\nSelect Sound Option:\n1. Mute (default)\n2. Sound (generate audio from pixel colors)\n");
                                                                                                                    if(tgtIsVid) printf("3. Target Sound (use audio from target video)\n");
                                                                                                                    while(true){
                                                                                                                        if(tgtIsVid) printf("Enter your choice (1-2, 3, default 1): ");
                                                                                                                        else printf("Enter your choice (1-2, default 1): ");
                                                                                                                        fflush(stdout);
                                                                                                                        std::string c;std::getline(std::cin,c);
                                                                                                                        if(c.empty()||c=="1") return{"mute",30};
                                                                                                                        if(c=="2") return{"sound",30};
                                                                                                                        if(c=="3"&&tgtIsVid) return selectTargetSoundQuality();
                                                                                                                        printf("Invalid choice.\n");
                                                                                                                    }
                                                                                                                }

                                                                                                                static cv::Mat processFramePair(const cv::Mat& baseBGR,const cv::Mat& tgtBGR,
                                                                                                                                                const std::string& algo,int resolution){
                                                                                                                    int limitRes=std::min({baseBGR.rows,baseBGR.cols,tgtBGR.rows,tgtBGR.cols});
                                                                                                                    int procRes=std::min(resolution,limitRes);
                                                                                                                    cv::Mat base,tgt;
                                                                                                                    cv::resize(baseBGR,base,cv::Size(procRes,procRes));
                                                                                                                    cv::resize(tgtBGR,tgt,cv::Size(procRes,procRes));
                                                                                                                    cv::cvtColor(base,base,cv::COLOR_BGR2RGB);
                                                                                                                    cv::cvtColor(tgt,tgt,cv::COLOR_BGR2RGB);
                                                                                                                    if(algo=="missform"){Missform m(base,tgt,127.f);return m.frame(1.f);}
                                                                                                                    std::vector<int32_t> asgn;
                                                                                                                    if(algo=="drawer") asgn=assignDrawerPixels(base,tgt);
                                                                                                                    else asgn=assignPixels(base,tgt,algo,cv::Mat());
                                                                                                                    int N=procRes*procRes,W=procRes;
                                                                                                                    cv::Mat frm(procRes,procRes,CV_8UC3,cv::Scalar(0,0,0));
                                                                                                                    const uint8_t* bp=base.data;
                                                                                                                    for(int i=0;i<N;i++){
                                                                                                                        if(asgn[i]<0) continue;
                                                                                                                        int dy=pyDiv(asgn[i],W),dx=pyMod(asgn[i],W);
                                                                                                                        dy=std::clamp(dy,0,procRes-1);dx=std::clamp(dx,0,procRes-1);
                                                                                                                        uint8_t* p=frm.data+(dy*W+dx)*3;
                                                                                                                        p[0]=bp[i*3];p[1]=bp[i*3+1];p[2]=bp[i*3+2];
                                                                                                                    }
                                                                                                                    return frm;
                                                                                                                                                }

                                                                                                                                                static void cliVideoProcess(const std::string& basePath,const std::string& tgtPath,
                                                                                                                                                                            const std::string& algo,int resolution,
                                                                                                                                                                            const std::string& soundOpt,int audioQuality){
                                                                                                                                                    const std::string outDir="results"; mkdir(outDir.c_str(),0755);
                                                                                                                                                    bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);

                                                                                                                                                    auto extractFrames=[&](const std::string& path,const char* desc,
                                                                                                                                                                           std::vector<cv::Mat>& frames,double& fps2){
                                                                                                                                                        cv::VideoCapture cap(path);
                                                                                                                                                        int total=(int)cap.get(cv::CAP_PROP_FRAME_COUNT);
                                                                                                                                                        fps2=cap.get(cv::CAP_PROP_FPS);
                                                                                                                                                        cv::Mat frm; int i=0;
                                                                                                                                                        while(cap.read(frm)){
                                                                                                                                                            frames.push_back(frm.clone());
                                                                                                                                                            i++;
                                                                                                                                                        }
                                                                                                                                                        cap.release();
                                                                                                                                                                           };

                                                                                                                                                                           std::vector<cv::Mat> bFrames,tFrames;
                                                                                                                                                                           double bFps=30,tFps=30;
                                                                                                                                                                           if(bIsVid) extractFrames(basePath,"Base",bFrames,bFps);
                                                                                                                                                                           else{
                                                                                                                                                                               cv::Mat img=cv::imread(basePath);
                                                                                                                                                                               if(img.empty()){fprintf(stderr,"Error: Could not load %s\n",basePath.c_str());exit(1);}
                                                                                                                                                                               bFrames.push_back(img);bFps=30;
                                                                                                                                                                           }
                                                                                                                                                                           if(tIsVid) extractFrames(tgtPath,"Target",tFrames,tFps);
                                                                                                                                                                           else{
                                                                                                                                                                               cv::Mat img=cv::imread(tgtPath);
                                                                                                                                                                               if(img.empty()){fprintf(stderr,"Error: Could not load %s\n",tgtPath.c_str());exit(1);}
                                                                                                                                                                               tFrames.push_back(img);tFps=30;
                                                                                                                                                                           }

                                                                                                                                                                           int total; double fps2,duration;
                                                                                                                                                                           if(bIsVid&&tIsVid){
                                                                                                                                                                               total=(int)std::min(bFrames.size(),tFrames.size());
                                                                                                                                                                               bFrames.resize(total);tFrames.resize(total);
                                                                                                                                                                               fps2=(bFrames.size()*bFps<=tFrames.size()*tFps)?bFps:tFps;
                                                                                                                                                                               duration=total/fps2;
                                                                                                                                                                           } else if(bIsVid){
                                                                                                                                                                               total=(int)bFrames.size();
                                                                                                                                                                               tFrames.assign(total,tFrames[0]);fps2=bFps;duration=total/fps2;
                                                                                                                                                                           } else if(tIsVid){
                                                                                                                                                                               total=(int)tFrames.size();
                                                                                                                                                                               bFrames.assign(total,bFrames[0]);fps2=tFps;duration=total/fps2;
                                                                                                                                                                           } else { total=1;fps2=30;duration=1.0; }

                                                                                                                                                                           std::vector<cv::Mat> processed;
                                                                                                                                                                           for(int i=0;i<total;i++){
                                                                                                                                                                               processed.push_back(processFramePair(bFrames[i],tFrames[i],algo,resolution));
                                                                                                                                                                           }

                                                                                                                                                                           time_t now2=time(nullptr);char ts2[32];
                                                                                                                                                                           strftime(ts2,sizeof(ts2),"%Y%m%d_%H%M%S",localtime(&now2));
                                                                                                                                                                           std::string videoPath,gifPath;
                                                                                                                                                                           gifPath=outDir+"/animation_"+ts2+".gif";

                                                                                                                                                                           if(soundOpt!="mute"){
                                                                                                                                                                               std::string silPath=outDir+"/video_"+ts2+"_silent.mp4";
                                                                                                                                                                               videoPath=outDir+"/video_"+ts2+".mp4";
                                                                                                                                                                               int fh=processed[0].rows,fw=processed[0].cols;
                                                                                                                                                                               cv::VideoWriter vw(silPath,cv::VideoWriter::fourcc('m','p','4','v'),fps2,cv::Size(fw,fh));
                                                                                                                                                                               for(int i=0;i<(int)processed.size();i++){
                                                                                                                                                                                   cv::Mat bgr;cv::cvtColor(processed[i],bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                                                                                                                                                                               }
                                                                                                                                                                               vw.release();
                                                                                                                                                                               std::string tgtAudio=(soundOpt=="target-sound"&&tIsVid)?tgtPath:"";
                                                                                                                                                                               std::string fin=addAudioToVideo(silPath,processed,fps2,videoPath,soundOpt,tgtAudio,audioQuality);
                                                                                                                                                                               remove(silPath.c_str());videoPath=fin;
                                                                                                                                                                           } else {
                                                                                                                                                                               videoPath=outDir+"/video_"+ts2+".mp4";
                                                                                                                                                                               int fh=processed[0].rows,fw=processed[0].cols;
                                                                                                                                                                               cv::VideoWriter vw(videoPath,cv::VideoWriter::fourcc('m','p','4','v'),fps2,cv::Size(fw,fh));
                                                                                                                                                                               for(int i=0;i<(int)processed.size();i++){
                                                                                                                                                                                   cv::Mat bgr;cv::cvtColor(processed[i],bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                                                                                                                                                                               }
                                                                                                                                                                               vw.release();
                                                                                                                                                                           }

                                                                                                                                                                           GIF::Encoder gifEnc; gifEnc.open(gifPath,processed[0].cols,processed[0].rows,1000/(int)fps2);
                                                                                                                                                                           for(int i=0;i<(int)processed.size();i++){
                                                                                                                                                                               gifEnc.writeFrame(processed[i]);
                                                                                                                                                                           }
                                                                                                                                                                           gifEnc.close();
                                                                                                                                                                            }

                                                                                                                                                                            static void cliImageProcess(const std::string& basePath,const std::string& tgtPath,
                                                                                                                                                                                                        const std::string& algo,int resolution,
                                                                                                                                                                                                        const std::string& soundOpt,int audioQuality){
                                                                                                                                                                                static const struct{const char* mode;const char* desc;} FMTS[]={
                                                                                                                                                                                    {"export_image","Frame (PNG)"},{"export_gif","GIF"},{"export_video","Animation (MP4)"}};
                                                                                                                                                                                    bool running=true;
                                                                                                                                                                                    for(auto& fmt:FMTS){
                                                                                                                                                                                        ProcessConfig cfg;
                                                                                                                                                                                        cfg.basePath=basePath;cfg.tgtPath=tgtPath;
                                                                                                                                                                                        cfg.mode=fmt.mode;cfg.algo=algo;
                                                                                                                                                                                        cfg.resolution=resolution;cfg.soundOpt=soundOpt;cfg.audioQuality=audioQuality;
                                                                                                                                                                                        cfg.fps=30;cfg.running=&running;
                                                                                                                                                                                        int lastPct=0;
                                                                                                                                                                                        processCore(cfg,
                                                                                                                                                                                                    [&](int v){},
                                                                                                                                                                                                    [](const QImage&){},
                                                                                                                                                                                                    [&](const std::string& m){},
                                                                                                                                                                                                    [&](const std::string& e){fprintf(stderr,"Error: %s\n",e.c_str());});
                                                                                                                                                                                    }
                                                                                                                                                                                                        }

                                                                                                                                                                                                        static void cliProcessAndExport(const std::string& basePath,const std::string& tgtPath,
                                                                                                                                                                                                                                        const std::string& algo,int resolution,
                                                                                                                                                                                                                                        const std::string& soundOpt,int audioQuality){
                                                                                                                                                                                                            bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);
                                                                                                                                                                                                            if(soundOpt=="target-sound"&&!tIsVid){
                                                                                                                                                                                                                fprintf(stderr,"Error: Target Sound option requires target to be a video file.\n");exit(1);}
                                                                                                                                                                                                                if(bIsVid||tIsVid){
                                                                                                                                                                                                                    if(algo=="fusion"){fprintf(stderr,"Error: Fusion algorithm cannot be used with video files.\n");exit(1);}
                                                                                                                                                                                                                    std::string a=algo;
                                                                                                                                                                                                                    if(a!="merge"&&a!="shuffle"&&a!="missform"){
                                                                                                                                                                                                                        a="merge";}
                                                                                                                                                                                                                        cliVideoProcess(basePath,tgtPath,a,resolution,soundOpt,audioQuality);
                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                    cliImageProcess(basePath,tgtPath,algo,resolution,soundOpt,audioQuality);
                                                                                                                                                                                                                }
                                                                                                                                                                                                                                        }

                                                                                                                                                                                                                                        static void interactiveCLI(){
                                                                                                                                                                                                                                            while(true){
                                                                                                                                                                                                                                                printBanner();
                                                                                                                                                                                                                                                printf("\n--- Media Selection ---\n");
                                                                                                                                                                                                                                                std::string basePath=getInputLine("Enter base media path (or drag & drop file): ",validateMediaFile);
                                                                                                                                                                                                                                                std::string tgtPath =getInputLine("Enter target media path (or drag & drop file): ",validateMediaFile);
                                                                                                                                                                                                                                                bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);
                                                                                                                                                                                                                                                std::string algo;
                                                                                                                                                                                                                                                if(bIsVid||tIsVid){
                                                                                                                                                                                                                                                    printf("\nVideo mode detected. Select algorithm:\n1. Merge (default)\n2. Shuffle\n3. Missform\n");
                                                                                                                                                                                                                                                    printf("Enter your choice (1-3, default 1): ");fflush(stdout);
                                                                                                                                                                                                                                                    std::string c;std::getline(std::cin,c);
                                                                                                                                                                                                                                                    if(c=="2") algo="shuffle"; else if(c=="3") algo="missform"; else algo="merge";
                                                                                                                                                                                                                                                } else algo=selectAlgorithm();
                                                                                                                                                                                                                                                int res=selectResolution();
                                                                                                                                                                                                                                                auto[sndOpt,sndQ]=selectSoundOption(tIsVid);
                                                                                                                                                                                                                                                cliProcessAndExport(basePath,tgtPath,algo,res,sndOpt,sndQ);
                                                                                                                                                                                                                                                printf("\n--- What's Next? ---\n1. Blend Again\n2. Exit\n");
                                                                                                                                                                                                                                                while(true){
                                                                                                                                                                                                                                                    printf("\nEnter your choice (1-2): ");fflush(stdout);
                                                                                                                                                                                                                                                    std::string c;std::getline(std::cin,c);
                                                                                                                                                                                                                                                    if(c=="1"){printf("\n============================================================\n\n");break;}
                                                                                                                                                                                                                                                    if(c=="2"){exit(0);}
                                                                                                                                                                                                                                                    printf("Invalid choice. Please enter 1 or 2.\n");
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        }

                                                                                                                                                                                                                                        class ProcessingThread : public QThread {
                                                                                                                                                                                                                                            Q_OBJECT
                                                                                                                                                                                                                                        public:
                                                                                                                                                                                                                                            ProcessConfig cfg;
                                                                                                                                                                                                                                            bool _running=true;
                                                                                                                                                                                                                                            explicit ProcessingThread(const ProcessConfig& c,QObject* p=nullptr):QThread(p),cfg(c){cfg.running=&_running;}
                                                                                                                                                                                                                                            void stop(){_running=false;}
                                                                                                                                                                                                                                        signals:
                                                                                                                                                                                                                                            void progressSignal(int);
                                                                                                                                                                                                                                            void frameSignal(QImage);
                                                                                                                                                                                                                                            void finishedSignal(QString);
                                                                                                                                                                                                                                            void errorSignal(QString);
                                                                                                                                                                                                                                        protected:
                                                                                                                                                                                                                                            void run() override {
                                                                                                                                                                                                                                                processCore(cfg,
                                                                                                                                                                                                                                                            [this](int v){emit progressSignal(v);},
                                                                                                                                                                                                                                                            [this](const QImage& q){emit frameSignal(q);},
                                                                                                                                                                                                                                                            [this](const std::string& m){emit finishedSignal(QString::fromStdString(m));},
                                                                                                                                                                                                                                                            [this](const std::string& e){emit errorSignal(QString::fromStdString(e));});
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                        class DrawingCanvas : public QLabel {
                                                                                                                                                                                                                                            Q_OBJECT
                                                                                                                                                                                                                                        public:
                                                                                                                                                                                                                                            bool drawing=false;
                                                                                                                                                                                                                                            QPoint lastPt;
                                                                                                                                                                                                                                            QColor penColor{0,0,0};
                                                                                                                                                                                                                                            int penWidth=5;
                                                                                                                                                                                                                                            QSize origSize{1024,1024};

                                                                                                                                                                                                                                            QImage baseQImage;
                                                                                                                                                                                                                                            QImage canvasLayer;
                                                                                                                                                                                                                                            std::vector<QImage> history;
                                                                                                                                                                                                                                            std::vector<QImage> redoStack;
                                                                                                                                                                                                                                            static constexpr int MAX_HIST=50;

                                                                                                                                                                                                                                            explicit DrawingCanvas(QWidget* parent=nullptr):QLabel(parent){
                                                                                                                                                                                                                                                setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
                                                                                                                                                                                                                                                setMouseTracking(true);
                                                                                                                                                                                                                                                canvasLayer=QImage(origSize,QImage::Format_ARGB32);
                                                                                                                                                                                                                                                canvasLayer.fill(Qt::transparent);
                                                                                                                                                                                                                                                setAlignment(Qt::AlignCenter);
                                                                                                                                                                                                                                                setStyleSheet("background-color:white;border:1px solid #404040;border-radius:4px;");
                                                                                                                                                                                                                                                setScaledContents(false);
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void setPenColor(const QColor& c){penColor=c;}
                                                                                                                                                                                                                                            void setPenWidth(int w){penWidth=std::max(1,std::min(50,w));}

                                                                                                                                                                                                                                            void updateDisplay(){
                                                                                                                                                                                                                                                QImage combined;
                                                                                                                                                                                                                                                if(!baseQImage.isNull()) combined=baseQImage.copy();
                                                                                                                                                                                                                                                else{ combined=QImage(origSize,QImage::Format_ARGB32); combined.fill(Qt::white); }
                                                                                                                                                                                                                                                QPainter p(&combined); p.drawImage(0,0,canvasLayer); p.end();
                                                                                                                                                                                                                                                if(!size().isEmpty()){
                                                                                                                                                                                                                                                    QImage scaled=combined.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
                                                                                                                                                                                                                                                    QLabel::setPixmap(QPixmap::fromImage(scaled));
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void resizeEvent(QResizeEvent* e) override { updateDisplay(); QLabel::resizeEvent(e); }

                                                                                                                                                                                                                                            QPoint getScaledPoint(const QPoint& pos){
                                                                                                                                                                                                                                                if(pixmap()&&!pixmap()->isNull()){
                                                                                                                                                                                                                                                    QSize ps=pixmap()->size(); QSize ls=size();
                                                                                                                                                                                                                                                    int ox=(ls.width()-ps.width())/2, oy=(ls.height()-ps.height())/2;
                                                                                                                                                                                                                                                    int sx=pos.x()-ox, sy=pos.y()-oy;
                                                                                                                                                                                                                                                    if(sx>=0&&sx<ps.width()&&sy>=0&&sy<ps.height()){
                                                                                                                                                                                                                                                        float scx=(float)origSize.width()/ps.width();
                                                                                                                                                                                                                                                        float scy=(float)origSize.height()/ps.height();
                                                                                                                                                                                                                                                        int cx=std::clamp((int)(sx*scx),0,origSize.width()-1);
                                                                                                                                                                                                                                                        int cy=std::clamp((int)(sy*scy),0,origSize.height()-1);
                                                                                                                                                                                                                                                        return {cx,cy};
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                return {-1,-1};
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void mousePressEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(e->button()==Qt::LeftButton){
                                                                                                                                                                                                                                                    drawing=true; QPoint p=getScaledPoint(e->pos()); if(p.x()>=0) lastPt=p; }
                                                                                                                                                                                                                                                    QLabel::mousePressEvent(e);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void mouseMoveEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(drawing){ QPoint p=getScaledPoint(e->pos()); if(p.x()>=0){drawLine(lastPt,p);lastPt=p;} }
                                                                                                                                                                                                                                                QLabel::mouseMoveEvent(e);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void mouseReleaseEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(e->button()==Qt::LeftButton&&drawing){drawing=false;saveToHistory();}
                                                                                                                                                                                                                                                QLabel::mouseReleaseEvent(e);
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void drawLine(const QPoint& a,const QPoint& b){
                                                                                                                                                                                                                                                QPainter p(&canvasLayer);
                                                                                                                                                                                                                                                p.setPen(QPen(penColor,penWidth,Qt::SolidLine,Qt::RoundCap,Qt::RoundJoin));
                                                                                                                                                                                                                                                p.drawLine(a,b); p.end(); updateDisplay();
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void saveToHistory(){
                                                                                                                                                                                                                                                if((int)history.size()>=MAX_HIST) history.erase(history.begin());
                                                                                                                                                                                                                                                history.push_back(canvasLayer.copy()); redoStack.clear();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            bool undo(){ if(history.size()>1){redoStack.push_back(history.back());history.pop_back();canvasLayer=history.back().copy();updateDisplay();return true;} return false; }
                                                                                                                                                                                                                                            bool redo(){ if(!redoStack.empty()){history.push_back(redoStack.back());redoStack.pop_back();canvasLayer=history.back().copy();updateDisplay();return true;} return false; }

                                                                                                                                                                                                                                            void clear(){
                                                                                                                                                                                                                                                canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
                                                                                                                                                                                                                                                history={canvasLayer.copy()}; redoStack.clear(); updateDisplay();
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            cv::Mat getImageArray(){
                                                                                                                                                                                                                                                QImage combined;
                                                                                                                                                                                                                                                if(!baseQImage.isNull()) combined=baseQImage.copy();
                                                                                                                                                                                                                                                else{ combined=QImage(origSize,QImage::Format_ARGB32); combined.fill(Qt::white); }
                                                                                                                                                                                                                                                QPainter p(&combined); p.drawImage(0,0,canvasLayer); p.end();
                                                                                                                                                                                                                                                QImage rgb=combined.convertToFormat(QImage::Format_RGB888);
                                                                                                                                                                                                                                                int w=rgb.width(),h=rgb.height();
                                                                                                                                                                                                                                                cv::Mat bgr(h,w,CV_8UC3);
                                                                                                                                                                                                                                                for(int y=0;y<h;y++){
                                                                                                                                                                                                                                                    const uint8_t* src=rgb.constScanLine(y);
                                                                                                                                                                                                                                                    uint8_t* dst=bgr.data+y*w*3;
                                                                                                                                                                                                                                                    for(int x=0;x<w;x++){dst[x*3]=src[x*3+2];dst[x*3+1]=src[x*3+1];dst[x*3+2]=src[x*3];}
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                return bgr;
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void setBaseImage(const cv::Mat& imgBGR){
                                                                                                                                                                                                                                                if(imgBGR.empty()){
                                                                                                                                                                                                                                                    baseQImage=QImage(); origSize=QSize(1024,1024);
                                                                                                                                                                                                                                                    canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
                                                                                                                                                                                                                                                    history.clear(); redoStack.clear();
                                                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                                                    int h=imgBGR.rows,w=imgBGR.cols; origSize=QSize(w,h);
                                                                                                                                                                                                                                                    cv::Mat rgb; cv::cvtColor(imgBGR,rgb,cv::COLOR_BGR2RGB);
                                                                                                                                                                                                                                                    QImage qi(rgb.data,w,h,w*3,QImage::Format_RGB888); baseQImage=qi.copy();
                                                                                                                                                                                                                                                    canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
                                                                                                                                                                                                                                                    history={canvasLayer.copy()}; redoStack.clear();
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                updateDisplay();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                        class ScalableImageLabel : public QLabel {
                                                                                                                                                                                                                                            Q_OBJECT
                                                                                                                                                                                                                                        public:
                                                                                                                                                                                                                                            QPixmap _pix;
                                                                                                                                                                                                                                            bool drawing=false;
                                                                                                                                                                                                                                            explicit ScalableImageLabel(QWidget* p=nullptr):QLabel(p){
                                                                                                                                                                                                                                                setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
                                                                                                                                                                                                                                                setAlignment(Qt::AlignCenter);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void setPixmap(const QPixmap& pm){ _pix=pm; updateDisplay(); }
                                                                                                                                                                                                                                            void resizeEvent(QResizeEvent* e) override { updateDisplay(); QLabel::resizeEvent(e); }
                                                                                                                                                                                                                                            void updateDisplay(){
                                                                                                                                                                                                                                                if(!_pix.isNull()) QLabel::setPixmap(_pix.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation));
                                                                                                                                                                                                                                                else QLabel::setPixmap(QPixmap());
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            std::pair<int,int> getCoords(const QPoint& pos){
                                                                                                                                                                                                                                                if(!_pix.isNull()){
                                                                                                                                                                                                                                                    QPixmap sp=_pix.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
                                                                                                                                                                                                                                                    int dx=(size().width()-sp.width())/2, dy=(size().height()-sp.height())/2;
                                                                                                                                                                                                                                                    int cx=pos.x()-dx, cy=pos.y()-dy;
                                                                                                                                                                                                                                                    if(cx>=0&&cx<sp.width()&&cy>=0&&cy<sp.height()){
                                                                                                                                                                                                                                                        float sx=(float)_pix.width()/sp.width();
                                                                                                                                                                                                                                                        float sy=(float)_pix.height()/sp.height();
                                                                                                                                                                                                                                                        return{(int)(cx*sx),(int)(cy*sy)};
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                return{-1,-1};
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void mousePressEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(e->button()==Qt::LeftButton){
                                                                                                                                                                                                                                                    drawing=true; auto[x,y]=getCoords(e->pos());
                                                                                                                                                                                                                                                    if(x>=0){emit clicked(x,y); emit drawn(x,y);}
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                QLabel::mousePressEvent(e);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void mouseMoveEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(drawing){auto[x,y]=getCoords(e->pos());if(x>=0)emit drawn(x,y);}
                                                                                                                                                                                                                                                QLabel::mouseMoveEvent(e);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void mouseReleaseEvent(QMouseEvent* e) override {
                                                                                                                                                                                                                                                if(e->button()==Qt::LeftButton){drawing=false;emit shapeCompleted();}
                                                                                                                                                                                                                                                QLabel::mouseReleaseEvent(e);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        signals:
                                                                                                                                                                                                                                            void clicked(int,int);
                                                                                                                                                                                                                                            void drawn(int,int);
                                                                                                                                                                                                                                            void shapeCompleted();
                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                        class MediaPanel : public QFrame {
                                                                                                                                                                                                                                            Q_OBJECT
                                                                                                                                                                                                                                        public:
                                                                                                                                                                                                                                            QString filePath, originalFilePath;
                                                                                                                                                                                                                                            bool isTarget=false;
                                                                                                                                                                                                                                            int rotateSteps=0; bool isFlipped=false;
                                                                                                                                                                                                                                            bool isAnalyzing=false;
                                                                                                                                                                                                                                            cv::Mat segments;
                                                                                                                                                                                                                                            std::set<int> selectedSegments;
                                                                                                                                                                                                                                            int procW=256,procH=256;
                                                                                                                                                                                                                                            QString penMode;
                                                                                                                                                                                                                                            std::vector<std::vector<std::pair<int,int>>> shapes;
                                                                                                                                                                                                                                            std::vector<std::pair<int,int>> currentShape;
                                                                                                                                                                                                                                            cv::Mat manualMask;
                                                                                                                                                                                                                                            ScalableImageLabel* preview=nullptr;
                                                                                                                                                                                                                                            DrawingCanvas* drawingCanvas=nullptr;
                                                                                                                                                                                                                                            QWidget* previewContainer=nullptr;
                                                                                                                                                                                                                                            QLabel* infoLbl=nullptr;
                                                                                                                                                                                                                                            QPushButton* analyzeBtn=nullptr,*penBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* addBtn=nullptr,*removeBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* rotateBtn=nullptr,*flipBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* clearBtn=nullptr,*resetBtn=nullptr;
                                                                                                                                                                                                                                            QWidget* drawerTools=nullptr;
                                                                                                                                                                                                                                            QPushButton* undoBtn=nullptr,*redoBtn=nullptr,*colorBtn=nullptr;
                                                                                                                                                                                                                                            QSlider* penSizeSlider=nullptr; QLabel* penSizeLbl=nullptr;
                                                                                                                                                                                                                                            bool drawerMode=false;

                                                                                                                                                                                                                                            explicit MediaPanel(const QString& title,bool isTgt=false,QWidget* p=nullptr)
                                                                                                                                                                                                                                            :QFrame(p),isTarget(isTgt){ setupUI(title); }

                                                                                                                                                                                                                                        signals:
                                                                                                                                                                                                                                            void mediaLoaded(QString);
                                                                                                                                                                                                                                            void mediaCleared();

                                                                                                                                                                                                                                        public slots:
                                                                                                                                                                                                                                            void loadMedia(){
                                                                                                                                                                                                                                                QString f=QFileDialog::getOpenFileName(this,"Open Media","","Images (*.png *.jpg *.jpeg *.webp)");
                                                                                                                                                                                                                                                if(!f.isEmpty()){ setMediaData(f,0,false); emit mediaLoaded(f); }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void clearMedia(){
                                                                                                                                                                                                                                                filePath.clear();originalFilePath.clear();rotateSteps=0;isFlipped=false;
                                                                                                                                                                                                                                                isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
                                                                                                                                                                                                                                                penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
                                                                                                                                                                                                                                                preview->setPixmap(QPixmap());infoLbl->setText("No media loaded");
                                                                                                                                                                                                                                                addBtn->setText("Add");removeBtn->setEnabled(false);
                                                                                                                                                                                                                                                rotateBtn->setEnabled(false);flipBtn->setEnabled(false);
                                                                                                                                                                                                                                                emit mediaCleared();
                                                                                                                                                                                                                                                if(drawingCanvas){ drawingCanvas->setBaseImage(cv::Mat()); drawingCanvas->clear(); }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void rotateMedia(){
                                                                                                                                                                                                                                                if(filePath.isEmpty()) return;
                                                                                                                                                                                                                                                rotateSteps=(rotateSteps+1)%4;updatePreview();
                                                                                                                                                                                                                                                if(drawerMode&&drawingCanvas) _reloadDrawerBase();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void flipMedia(){
                                                                                                                                                                                                                                                if(filePath.isEmpty()) return;
                                                                                                                                                                                                                                                isFlipped=!isFlipped;updatePreview();
                                                                                                                                                                                                                                                if(drawerMode&&drawingCanvas) _reloadDrawerBase();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void analyzeShapes(){
                                                                                                                                                                                                                                                if(filePath.isEmpty()) return;
                                                                                                                                                                                                                                                try{
                                                                                                                                                                                                                                                    cv::Mat img=cv::imread(filePath.toStdString());
                                                                                                                                                                                                                                                    img=applyTransforms(img,rotateSteps,isFlipped);
                                                                                                                                                                                                                                                    int h=img.rows,w=img.cols;
                                                                                                                                                                                                                                                    if(!penMode.isEmpty()&&(!shapes.empty()||!currentShape.empty())){
                                                                                                                                                                                                                                                        if(!currentShape.empty()){shapes.push_back(currentShape);currentShape.clear();}
                                                                                                                                                                                                                                                        cv::Mat mask=(penMode=="plus")?cv::Mat::zeros(h,w,CV_8U):cv::Mat(h,w,CV_8U,cv::Scalar(255));
                                                                                                                                                                                                                                                        uint8_t col=(penMode=="plus")?255:0;
                                                                                                                                                                                                                                                        for(auto& shape:shapes){
                                                                                                                                                                                                                                                            if((int)shape.size()>=3){
                                                                                                                                                                                                                                                                std::vector<cv::Point> pts;
                                                                                                                                                                                                                                                                for(auto[x,y]:shape) pts.push_back({x,y});
                                                                                                                                                                                                                                                                std::vector<std::vector<cv::Point>> contours={pts};
                                                                                                                                                                                                                                                                cv::fillPoly(mask,contours,cv::Scalar(col));
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                        manualMask=(mask>0);
                                                                                                                                                                                                                                                        isAnalyzing=true;updatePreview();
                                                                                                                                                                                                                                                        int sc=(int)shapes.size();
                                                                                                                                                                                                                                                        if(sc<=1) infoLbl->setText("Shape analyzed.");
                                                                                                                                                                                                                                                        else infoLbl->setText(QString("%1 shapes analyzed (combined mask, visually separate).").arg(sc));
                                                                                                                                                                                                                                                        return;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    procW=256;procH=256;
                                                                                                                                                                                                                                                    cv::Mat small;cv::resize(img,small,cv::Size(procW,procH));
                                                                                                                                                                                                                                                    cv::Mat blurred; cv::blur(small,blurred,cv::Size(5,5));
                                                                                                                                                                                                                                                    cv::Mat rgb;cv::cvtColor(blurred,rgb,cv::COLOR_BGR2RGB);
                                                                                                                                                                                                                                                    cv::Mat pv=rgb.reshape(1,procW*procH);
                                                                                                                                                                                                                                                    pv.convertTo(pv,CV_32F);
                                                                                                                                                                                                                                                    cv::Mat labels,centers;
                                                                                                                                                                                                                                                    cv::TermCriteria crit(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,10,1.0);
                                                                                                                                                                                                                                                    cv::kmeans(pv,6,labels,crit,10,cv::KMEANS_RANDOM_CENTERS,centers);
                                                                                                                                                                                                                                                    segments=labels.reshape(1,procH);
                                                                                                                                                                                                                                                    selectedSegments.clear();isAnalyzing=true;
                                                                                                                                                                                                                                                    updatePreview();infoLbl->setText("Click shape to select (Green), others (Red)");
                                                                                                                                                                                                                                                }catch(const std::exception& e){ QMessageBox::warning(this,"Analysis Error",e.what()); }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void stopAnalysis(){
                                                                                                                                                                                                                                                isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
                                                                                                                                                                                                                                                penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
                                                                                                                                                                                                                                                updatePreview();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void setPenMode(const QString& m){
                                                                                                                                                                                                                                                penMode=m;shapes.clear();currentShape.clear();manualMask=cv::Mat();
                                                                                                                                                                                                                                                isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();updatePreview();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void undoDrawing(){ if(drawingCanvas) drawingCanvas->undo(); }
                                                                                                                                                                                                                                            void redoDrawing(){ if(drawingCanvas) drawingCanvas->redo(); }
                                                                                                                                                                                                                                            void pickColor(){
                                                                                                                                                                                                                                                if(!drawingCanvas) return;
                                                                                                                                                                                                                                                QColor c=QColorDialog::getColor(drawingCanvas->penColor,this);
                                                                                                                                                                                                                                                if(c.isValid()) drawingCanvas->setPenColor(c);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void updatePenSize(int v){
                                                                                                                                                                                                                                                if(drawingCanvas) drawingCanvas->setPenWidth(v);
                                                                                                                                                                                                                                                penSizeLbl->setText(QString("Size: %1").arg(v));
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void clearDrawing(){ if(drawingCanvas) drawingCanvas->clear(); }
                                                                                                                                                                                                                                            void resetCanvas(){ if(drawingCanvas){drawingCanvas->setBaseImage(cv::Mat());drawingCanvas->clear();} }

                                                                                                                                                                                                                                            void onPreviewClicked(int x,int y){
                                                                                                                                                                                                                                                if(!isAnalyzing) return;
                                                                                                                                                                                                                                                if(!manualMask.empty()) return;
                                                                                                                                                                                                                                                if(segments.empty()||preview->_pix.isNull()) return;
                                                                                                                                                                                                                                                int ow=preview->_pix.width(),oh=preview->_pix.height();
                                                                                                                                                                                                                                                int sx=std::clamp((int)(x*(float)procW/ow),0,procW-1);
                                                                                                                                                                                                                                                int sy=std::clamp((int)(y*(float)procH/oh),0,procH-1);
                                                                                                                                                                                                                                                int lbl=segments.at<int>(sy,sx);
                                                                                                                                                                                                                                                if(selectedSegments.count(lbl)) selectedSegments.erase(lbl);
                                                                                                                                                                                                                                                else selectedSegments.insert(lbl);
                                                                                                                                                                                                                                                updatePreview();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void onPreviewDrawn(int x,int y){
                                                                                                                                                                                                                                                if(!penMode.isEmpty()&&!isAnalyzing){ currentShape.push_back({x,y}); updatePreview(); }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void onShapeCompleted(){
                                                                                                                                                                                                                                                if(!penMode.isEmpty()&&!isAnalyzing&&!currentShape.empty()){
                                                                                                                                                                                                                                                    shapes.push_back(currentShape); currentShape.clear();
                                                                                                                                                                                                                                                    infoLbl->setText(QString("Shape %1 completed. Draw another or click Analyze.").arg((int)shapes.size()));
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void setDrawerMode(bool enabled){
                                                                                                                                                                                                                                                bool was=drawerMode; drawerMode=enabled;
                                                                                                                                                                                                                                                if(enabled){
                                                                                                                                                                                                                                                    if(!drawingCanvas){ drawingCanvas=new DrawingCanvas(); }
                                                                                                                                                                                                                                                    auto* lo=qobject_cast<QVBoxLayout*>(previewContainer->layout());
                                                                                                                                                                                                                                                    lo->replaceWidget(preview,drawingCanvas);
                                                                                                                                                                                                                                                    preview->setVisible(false); drawingCanvas->setVisible(true);
                                                                                                                                                                                                                                                    drawingCanvas->clear();
                                                                                                                                                                                                                                                    addBtn->setVisible(false);removeBtn->setVisible(false);
                                                                                                                                                                                                                                                    rotateBtn->setVisible(false);flipBtn->setVisible(false);
                                                                                                                                                                                                                                                    clearBtn->setVisible(true);resetBtn->setVisible(true);
                                                                                                                                                                                                                                                    drawerTools->setVisible(true);infoLbl->setText("Draw on canvas");
                                                                                                                                                                                                                                                    if(!filePath.isEmpty()) _reloadDrawerBase();
                                                                                                                                                                                                                                                    else drawingCanvas->setBaseImage(cv::Mat());
                                                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                                                    if(was&&drawingCanvas){
                                                                                                                                                                                                                                                        cv::Mat drawing=drawingCanvas->getImageArray();
                                                                                                                                                                                                                                                        if(!drawing.empty()){
                                                                                                                                                                                                                                                            std::string tmp="/tmp/imder_draw_"+std::to_string((uintptr_t)this)+"_"+
                                                                                                                                                                                                                                                            std::to_string((int)time(nullptr))+".png";
                                                                                                                                                                                                                                                            cv::imwrite(tmp,drawing);
                                                                                                                                                                                                                                                            filePath=QString::fromStdString(tmp);
                                                                                                                                                                                                                                                            originalFilePath=filePath;
                                                                                                                                                                                                                                                            addBtn->setText("Replace");
                                                                                                                                                                                                                                                            updatePreview();
                                                                                                                                                                                                                                                            infoLbl->setText(QString("Drawing (%1x%2)").arg(drawing.cols).arg(drawing.rows));
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                        auto* lo=qobject_cast<QVBoxLayout*>(previewContainer->layout());
                                                                                                                                                                                                                                                        lo->replaceWidget(drawingCanvas,preview);
                                                                                                                                                                                                                                                        drawingCanvas->setVisible(false);preview->setVisible(true);
                                                                                                                                                                                                                                                        preview->updateDisplay();
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    addBtn->setVisible(true);removeBtn->setVisible(true);
                                                                                                                                                                                                                                                    rotateBtn->setVisible(true);flipBtn->setVisible(true);
                                                                                                                                                                                                                                                    clearBtn->setVisible(false);resetBtn->setVisible(false);
                                                                                                                                                                                                                                                    drawerTools->setVisible(false);
                                                                                                                                                                                                                                                    if(filePath.isEmpty()) infoLbl->setText("No media loaded");
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            cv::Mat getMask(){
                                                                                                                                                                                                                                                if(!manualMask.empty()) return manualMask;
                                                                                                                                                                                                                                                if(segments.empty()||selectedSegments.empty()) return cv::Mat();
                                                                                                                                                                                                                                                cv::Mat mask(procH,procW,CV_8U,cv::Scalar(0));
                                                                                                                                                                                                                                                for(int y=0;y<procH;y++) for(int x=0;x<procW;x++)
                                                                                                                                                                                                                                                    if(selectedSegments.count(segments.at<int>(y,x))) mask.at<uint8_t>(y,x)=255;
                                                                                                                                                                                                                                                    return mask;
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            cv::Mat getDrawingArray(){ return drawingCanvas?drawingCanvas->getImageArray():cv::Mat(); }

                                                                                                                                                                                                                                            void setMediaData(const QString& path,int rot,bool flip){
                                                                                                                                                                                                                                                originalFilePath=path; filePath=path;
                                                                                                                                                                                                                                                rotateSteps=rot; isFlipped=flip;
                                                                                                                                                                                                                                                isAnalyzing=false;segments=cv::Mat();
                                                                                                                                                                                                                                                penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
                                                                                                                                                                                                                                                if(!path.isEmpty()){
                                                                                                                                                                                                                                                    cv::Mat img=cv::imread(path.toStdString());
                                                                                                                                                                                                                                                    if(!img.empty()){
                                                                                                                                                                                                                                                        QFileInfo fi(path); double sz=(double)fi.size()/(1024*1024);
                                                                                                                                                                                                                                                        infoLbl->setText(QString("%1 (%2 MB) %3x%4").arg(fi.fileName()).arg(sz,0,'f',1).arg(img.cols).arg(img.rows));
                                                                                                                                                                                                                                                    } else {
                                                                                                                                                                                                                                                        QFileInfo fi(path); double sz=(double)fi.size()/(1024*1024);
                                                                                                                                                                                                                                                        infoLbl->setText(QString("%1 (%2 MB)").arg(fi.fileName()).arg(sz,0,'f',1));
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    addBtn->setText("Replace"); removeBtn->setEnabled(true);
                                                                                                                                                                                                                                                    rotateBtn->setEnabled(true); flipBtn->setEnabled(true);
                                                                                                                                                                                                                                                    updatePreview();
                                                                                                                                                                                                                                                    if(drawerMode&&drawingCanvas) _reloadDrawerBase();
                                                                                                                                                                                                                                                } else clearMedia();
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void updatePreview(){
                                                                                                                                                                                                                                                if(filePath.isEmpty()) return;
                                                                                                                                                                                                                                                try{
                                                                                                                                                                                                                                                    cv::Mat img=cv::imread(filePath.toStdString());
                                                                                                                                                                                                                                                    if(img.empty()){infoLbl->setText("Error loading image");return;}
                                                                                                                                                                                                                                                    img=applyTransforms(img,rotateSteps,isFlipped);
                                                                                                                                                                                                                                                    if(isAnalyzing){
                                                                                                                                                                                                                                                        if(!manualMask.empty()){
                                                                                                                                                                                                                                                            cv::Mat viz=img.clone();
                                                                                                                                                                                                                                                            cv::Mat overlay=cv::Mat::zeros(img.size(),img.type());
                                                                                                                                                                                                                                                            cv::Mat maskFull;
                                                                                                                                                                                                                                                            if(manualMask.rows==img.rows&&manualMask.cols==img.cols) maskFull=manualMask;
                                                                                                                                                                                                                                                            else cv::resize(manualMask,maskFull,img.size(),0,0,cv::INTER_NEAREST);
                                                                                                                                                                                                                                                            for(int y=0;y<img.rows;y++) for(int x=0;x<img.cols;x++){
                                                                                                                                                                                                                                                                bool m=maskFull.at<uint8_t>(y,x)>0;
                                                                                                                                                                                                                                                                overlay.at<cv::Vec3b>(y,x)=m?cv::Vec3b(0,255,0):cv::Vec3b(255,0,0);
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                            cv::addWeighted(viz,0.7,overlay,0.3,0,viz);img=viz;
                                                                                                                                                                                                                                                        } else if(!segments.empty()){
                                                                                                                                                                                                                                                            cv::Mat ovl;cv::resize(img,ovl,cv::Size(procW,procH));
                                                                                                                                                                                                                                                            cv::Mat viz=ovl.clone();
                                                                                                                                                                                                                                                            auto uq=std::set<int>();
                                                                                                                                                                                                                                                            for(int y=0;y<procH;y++) for(int x=0;x<procW;x++) uq.insert(segments.at<int>(y,x));
                                                                                                                                                                                                                                                            for(int lbl:uq){
                                                                                                                                                                                                                                                                cv::Mat m=cv::Mat::zeros(procH,procW,CV_8U);
                                                                                                                                                                                                                                                                for(int y=0;y<procH;y++) for(int x=0;x<procW;x++)
                                                                                                                                                                                                                                                                if(segments.at<int>(y,x)==lbl) m.at<uint8_t>(y,x)=255;
                                                                                                                                                                                                                                                                std::vector<std::vector<cv::Point>> cnts;
                                                                                                                                                                                                                                                                cv::findContours(m,cnts,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                                                                                                                                                                                                                                                                bool sel=selectedSegments.count(lbl)>0;
                                                                                                                                                                                                                                                                if(selectedSegments.empty()) cv::drawContours(viz,cnts,-1,cv::Scalar(255,255,0),2);
                                                                                                                                                                                                                                                                else if(sel) cv::drawContours(viz,cnts,-1,cv::Scalar(0,255,0),2);
                                                                                                                                                                                                                                                                else cv::drawContours(viz,cnts,-1,cv::Scalar(0,0,255),1);
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                            img=viz;
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                    } else if(!penMode.isEmpty()&&(!shapes.empty()||!currentShape.empty())){
                                                                                                                                                                                                                                                        cv::Mat viz=img.clone();
                                                                                                                                                                                                                                                        cv::Scalar col=(penMode=="plus")?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
                                                                                                                                                                                                                                                        for(auto& sh:shapes){
                                                                                                                                                                                                                                                            if((int)sh.size()>=2){
                                                                                                                                                                                                                                                                std::vector<cv::Point> pts;for(auto[x,y]:sh)pts.push_back({x,y});
                                                                                                                                                                                                                                                                std::vector<std::vector<cv::Point>> c={pts};
                                                                                                                                                                                                                                                                cv::polylines(viz,c,false,col,2);
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                        if((int)currentShape.size()>=2){
                                                                                                                                                                                                                                                            std::vector<cv::Point> pts;for(auto[x,y]:currentShape)pts.push_back({x,y});
                                                                                                                                                                                                                                                            std::vector<std::vector<cv::Point>> c={pts};
                                                                                                                                                                                                                                                            cv::polylines(viz,c,false,col,2);
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                        img=viz;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    cv::Mat rgb;cv::cvtColor(img,rgb,cv::COLOR_BGR2RGB);
                                                                                                                                                                                                                                                    int w=rgb.cols,h=rgb.rows;
                                                                                                                                                                                                                                                    QImage qi(rgb.data,w,h,w*3,QImage::Format_RGB888);
                                                                                                                                                                                                                                                    preview->setPixmap(QPixmap::fromImage(qi.copy()));
                                                                                                                                                                                                                                                }catch(const std::exception& e){infoLbl->setText(QString("Error: ")+e.what());}
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                        private:
                                                                                                                                                                                                                                            void _reloadDrawerBase(){
                                                                                                                                                                                                                                                cv::Mat img=cv::imread(filePath.toStdString());
                                                                                                                                                                                                                                                if(!img.empty()){ img=applyTransforms(img,rotateSteps,isFlipped); drawingCanvas->setBaseImage(img); }
                                                                                                                                                                                                                                                else drawingCanvas->setBaseImage(cv::Mat());
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void setupUI(const QString& title){
                                                                                                                                                                                                                                                setStyleSheet(panelStyle()); setObjectName("mediaPanel");
                                                                                                                                                                                                                                                auto* lo=new QVBoxLayout(this); lo->setContentsMargins(12,12,12,12); lo->setSpacing(10);
                                                                                                                                                                                                                                                auto* tlbl=new QLabel(title); tlbl->setStyleSheet(titleLblStyle()); tlbl->setAlignment(Qt::AlignCenter);
                                                                                                                                                                                                                                                lo->addWidget(tlbl);
                                                                                                                                                                                                                                                auto* toolsLo=new QHBoxLayout();
                                                                                                                                                                                                                                                analyzeBtn=new QPushButton("Analyze Shapes");
                                                                                                                                                                                                                                                analyzeBtn->setStyleSheet(surfBtnStyle()+"border:1px solid #FFEB3B;color:#FFEB3B;");
                                                                                                                                                                                                                                                analyzeBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                connect(analyzeBtn,&QPushButton::clicked,this,&MediaPanel::analyzeShapes);
                                                                                                                                                                                                                                                analyzeBtn->setVisible(false);
                                                                                                                                                                                                                                                penBtn=new QPushButton("Pen");
                                                                                                                                                                                                                                                penBtn->setStyleSheet(surfBtnStyle());
                                                                                                                                                                                                                                                penBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                auto* penMenu=new QMenu();
                                                                                                                                                                                                                                                penMenu->setStyleSheet(menuStyle());
                                                                                                                                                                                                                                                penMenu->addAction("+ (Include)",[this]{setPenMode("plus");});
                                                                                                                                                                                                                                                penMenu->addAction("- (Exclude)",[this]{setPenMode("minus");});
                                                                                                                                                                                                                                                penBtn->setMenu(penMenu); penBtn->setVisible(false);
                                                                                                                                                                                                                                                toolsLo->addWidget(analyzeBtn); toolsLo->addWidget(penBtn);
                                                                                                                                                                                                                                                previewContainer=new QWidget();
                                                                                                                                                                                                                                                auto* pvLo=new QVBoxLayout(previewContainer); pvLo->setContentsMargins(0,0,0,0);
                                                                                                                                                                                                                                                preview=new ScalableImageLabel();
                                                                                                                                                                                                                                                preview->setStyleSheet(previewLblStyle()); preview->setMinimumSize(200,200);
                                                                                                                                                                                                                                                connect(preview,&ScalableImageLabel::clicked,this,&MediaPanel::onPreviewClicked);
                                                                                                                                                                                                                                                connect(preview,&ScalableImageLabel::drawn,this,&MediaPanel::onPreviewDrawn);
                                                                                                                                                                                                                                                connect(preview,&ScalableImageLabel::shapeCompleted,this,&MediaPanel::onShapeCompleted);
                                                                                                                                                                                                                                                pvLo->addWidget(preview);
                                                                                                                                                                                                                                                lo->addLayout(toolsLo); lo->addWidget(previewContainer,1);
                                                                                                                                                                                                                                                infoLbl=new QLabel("No media loaded"); infoLbl->setStyleSheet(subtitleLblStyle());
                                                                                                                                                                                                                                                infoLbl->setAlignment(Qt::AlignCenter); lo->addWidget(infoLbl);
                                                                                                                                                                                                                                                drawerTools=new QWidget(); auto* dlo=new QGridLayout(drawerTools);
                                                                                                                                                                                                                                                dlo->setContentsMargins(0,0,0,0); dlo->setSpacing(5);
                                                                                                                                                                                                                                                undoBtn=new QPushButton("Undo");undoBtn->setStyleSheet(surfBtnStyle());undoBtn->setCursor(Qt::PointingHandCursor);connect(undoBtn,&QPushButton::clicked,this,&MediaPanel::undoDrawing);
                                                                                                                                                                                                                                                redoBtn=new QPushButton("Redo");redoBtn->setStyleSheet(surfBtnStyle());redoBtn->setCursor(Qt::PointingHandCursor);connect(redoBtn,&QPushButton::clicked,this,&MediaPanel::redoDrawing);
                                                                                                                                                                                                                                                colorBtn=new QPushButton("Color");colorBtn->setStyleSheet(surfBtnStyle());colorBtn->setCursor(Qt::PointingHandCursor);connect(colorBtn,&QPushButton::clicked,this,&MediaPanel::pickColor);
                                                                                                                                                                                                                                                penSizeLbl=new QLabel("Size: 5");penSizeLbl->setStyleSheet(subtitleLblStyle());
                                                                                                                                                                                                                                                penSizeSlider=new QSlider(Qt::Horizontal);penSizeSlider->setRange(1,50);penSizeSlider->setValue(5);
                                                                                                                                                                                                                                                penSizeSlider->setStyleSheet(sliderStyle());
                                                                                                                                                                                                                                                connect(penSizeSlider,&QSlider::valueChanged,this,&MediaPanel::updatePenSize);
                                                                                                                                                                                                                                                dlo->addWidget(undoBtn,0,0);dlo->addWidget(redoBtn,0,1);dlo->addWidget(colorBtn,0,2);
                                                                                                                                                                                                                                                dlo->addWidget(penSizeLbl,1,0);dlo->addWidget(penSizeSlider,1,1,1,2);
                                                                                                                                                                                                                                                drawerTools->setVisible(false); lo->addWidget(drawerTools);
                                                                                                                                                                                                                                                auto* btnLo=new QHBoxLayout();
                                                                                                                                                                                                                                                addBtn=new QPushButton("Add");addBtn->setStyleSheet(mainBtnStyle());addBtn->setCursor(Qt::PointingHandCursor);connect(addBtn,&QPushButton::clicked,this,&MediaPanel::loadMedia);
                                                                                                                                                                                                                                                removeBtn=new QPushButton("Remove");removeBtn->setStyleSheet(surfBtnStyle());removeBtn->setCursor(Qt::PointingHandCursor);connect(removeBtn,&QPushButton::clicked,this,&MediaPanel::clearMedia);removeBtn->setEnabled(false);
                                                                                                                                                                                                                                                btnLo->addWidget(addBtn);btnLo->addWidget(removeBtn);lo->addLayout(btnLo);
                                                                                                                                                                                                                                                auto* manipLo=new QHBoxLayout();
                                                                                                                                                                                                                                                rotateBtn=new QPushButton("Rotate");rotateBtn->setStyleSheet(surfBtnStyle());rotateBtn->setCursor(Qt::PointingHandCursor);connect(rotateBtn,&QPushButton::clicked,this,&MediaPanel::rotateMedia);rotateBtn->setEnabled(false);
                                                                                                                                                                                                                                                flipBtn=new QPushButton("Flip");flipBtn->setStyleSheet(surfBtnStyle());flipBtn->setCursor(Qt::PointingHandCursor);connect(flipBtn,&QPushButton::clicked,this,&MediaPanel::flipMedia);flipBtn->setEnabled(false);
                                                                                                                                                                                                                                                clearBtn=new QPushButton("Clear");clearBtn->setStyleSheet(surfBtnStyle());clearBtn->setCursor(Qt::PointingHandCursor);connect(clearBtn,&QPushButton::clicked,this,&MediaPanel::clearDrawing);clearBtn->setVisible(false);
                                                                                                                                                                                                                                                resetBtn=new QPushButton("Reset");resetBtn->setStyleSheet(surfBtnStyle());resetBtn->setCursor(Qt::PointingHandCursor);connect(resetBtn,&QPushButton::clicked,this,&MediaPanel::resetCanvas);resetBtn->setVisible(false);
                                                                                                                                                                                                                                                manipLo->addWidget(rotateBtn);manipLo->addWidget(flipBtn);manipLo->addWidget(resetBtn);manipLo->addWidget(clearBtn);
                                                                                                                                                                                                                                                lo->addLayout(manipLo);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                        class ImderGUI : public QMainWindow {
                                                                                                                                                                                                                                            Q_OBJECT
                                                                                                                                                                                                                                        public:
                                                                                                                                                                                                                                            QComboBox* modeCombo=nullptr;
                                                                                                                                                                                                                                            QComboBox* resCombo=nullptr;
                                                                                                                                                                                                                                            QComboBox* fpsCombo=nullptr;
                                                                                                                                                                                                                                            MediaPanel* basePanel=nullptr;
                                                                                                                                                                                                                                            MediaPanel* targetPanel=nullptr;
                                                                                                                                                                                                                                            QFrame* previewPanel=nullptr;
                                                                                                                                                                                                                                            ScalableImageLabel* previewDisplay=nullptr;
                                                                                                                                                                                                                                            QPushButton* reverseBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* startBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* replayBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* stopBtn=nullptr;
                                                                                                                                                                                                                                            QPushButton* exportBtn=nullptr;
                                                                                                                                                                                                                                            QLabel* statusBar=nullptr;
                                                                                                                                                                                                                                            QProgressBar* progress=nullptr;
                                                                                                                                                                                                                                            ProcessingThread* worker=nullptr;
                                                                                                                                                                                                                                            std::vector<QImage> cachedFrames;
                                                                                                                                                                                                                                            std::vector<QImage> replayFrames;
                                                                                                                                                                                                                                            QTimer* replayTimer=nullptr;
                                                                                                                                                                                                                                            qint64 replayStartMs=0;
                                                                                                                                                                                                                                            double replayDuration=10.0;
                                                                                                                                                                                                                                            int currentFps=30;

                                                                                                                                                                                                                                            explicit ImderGUI(QWidget* p=nullptr):QMainWindow(p){
                                                                                                                                                                                                                                                setWindowTitle("Imder - Image Blender");
                                                                                                                                                                                                                                                resize(1100,750);
                                                                                                                                                                                                                                                setStyleSheet(windowStyle());
                                                                                                                                                                                                                                                QString iconPath=QCoreApplication::applicationDirPath()+"/imder.png";
                                                                                                                                                                                                                                                if(QFile::exists(iconPath)){QIcon ic(iconPath);setWindowIcon(ic);QApplication::setWindowIcon(ic);}

                                                                                                                                                                                                                                                auto* central=new QWidget(); setCentralWidget(central);
                                                                                                                                                                                                                                                auto* mainLo=new QVBoxLayout(central); mainLo->setContentsMargins(16,16,16,16); mainLo->setSpacing(16);

                                                                                                                                                                                                                                                auto* headerLo=new QHBoxLayout(); headerLo->setSpacing(12);
                                                                                                                                                                                                                                                auto* modeLbl=new QLabel("Mode:"); modeLbl->setStyleSheet(subtitleLblStyle());
                                                                                                                                                                                                                                                modeCombo=new QComboBox();
                                                                                                                                                                                                                                                modeCombo->addItems({"Shuffle","Merge","Missform","Fusion","Pattern","Disguise","Navigate","Swap","Blend","Drawer"});
                                                                                                                                                                                                                                                modeCombo->setToolTip("Select processing mode. All modes work with Images.");
                                                                                                                                                                                                                                                modeCombo->setStyleSheet(comboStyle());
                                                                                                                                                                                                                                                connect(modeCombo,&QComboBox::currentTextChanged,this,&ImderGUI::onModeChanged);
                                                                                                                                                                                                                                                auto* resLbl=new QLabel("Resolution:"); resLbl->setStyleSheet(subtitleLblStyle());
                                                                                                                                                                                                                                                resCombo=new QComboBox();
                                                                                                                                                                                                                                                resCombo->addItems({"128x128","256x256","512x512","768x768","1024x1024","2048x2048","Custom"});
                                                                                                                                                                                                                                                resCombo->setStyleSheet(comboStyle()); resCombo->setMinimumWidth(120);
                                                                                                                                                                                                                                                connect(resCombo,&QComboBox::currentTextChanged,this,&ImderGUI::onResolutionChanged);
                                                                                                                                                                                                                                                auto* fpsLbl=new QLabel("FPS:"); fpsLbl->setStyleSheet(subtitleLblStyle());
                                                                                                                                                                                                                                                fpsCombo=new QComboBox();
                                                                                                                                                                                                                                                fpsCombo->addItems({"30","60","90","120","240"});
                                                                                                                                                                                                                                                fpsCombo->setCurrentIndex(0); fpsCombo->setStyleSheet(comboStyle()); fpsCombo->setMinimumWidth(80);
                                                                                                                                                                                                                                                headerLo->addWidget(modeLbl);headerLo->addWidget(modeCombo);
                                                                                                                                                                                                                                                headerLo->addWidget(resLbl);headerLo->addWidget(resCombo);
                                                                                                                                                                                                                                                headerLo->addWidget(fpsLbl);headerLo->addWidget(fpsCombo);
                                                                                                                                                                                                                                                headerLo->addStretch();
                                                                                                                                                                                                                                                mainLo->addLayout(headerLo);

                                                                                                                                                                                                                                                auto* panelsLo=new QHBoxLayout(); panelsLo->setSpacing(16);
                                                                                                                                                                                                                                                basePanel=new MediaPanel("Base",false);
                                                                                                                                                                                                                                                previewPanel=new QFrame(); previewPanel->setStyleSheet(panelStyle());
                                                                                                                                                                                                                                                auto* plo=new QVBoxLayout(previewPanel); plo->setContentsMargins(12,12,12,12); plo->setSpacing(10);
                                                                                                                                                                                                                                                reverseBtn=new QPushButton("Swap"); reverseBtn->setStyleSheet(surfBtnStyle());
                                                                                                                                                                                                                                                reverseBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                connect(reverseBtn,&QPushButton::clicked,this,&ImderGUI::swapMedia);
                                                                                                                                                                                                                                                reverseBtn->setEnabled(false); plo->addWidget(reverseBtn);
                                                                                                                                                                                                                                                auto* pLbl=new QLabel("Animation Preview"); pLbl->setStyleSheet(titleLblStyle()); pLbl->setAlignment(Qt::AlignCenter);
                                                                                                                                                                                                                                                plo->addWidget(pLbl);
                                                                                                                                                                                                                                                previewDisplay=new ScalableImageLabel();
                                                                                                                                                                                                                                                previewDisplay->setStyleSheet(previewLblStyle()); previewDisplay->setMinimumSize(200,200);
                                                                                                                                                                                                                                                plo->addWidget(previewDisplay,1);
                                                                                                                                                                                                                                                targetPanel=new MediaPanel("Target",true); targetPanel->setEnabled(false);
                                                                                                                                                                                                                                                panelsLo->addWidget(basePanel,1); panelsLo->addWidget(previewPanel,1); panelsLo->addWidget(targetPanel,1);
                                                                                                                                                                                                                                                mainLo->addLayout(panelsLo);

                                                                                                                                                                                                                                                auto* ctrlLo=new QHBoxLayout(); ctrlLo->setSpacing(12);
                                                                                                                                                                                                                                                startBtn=new QPushButton("Start Processing"); startBtn->setStyleSheet(mainBtnStyle()); startBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                replayBtn=new QPushButton("Replay"); replayBtn->setStyleSheet(mainBtnStyle()); replayBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                auto* replayMenu=new QMenu(); replayMenu->setStyleSheet(menuStyle());
                                                                                                                                                                                                                                                replayMenu->addAction("Show Animate",[this]{runReplay(false);});
                                                                                                                                                                                                                                                replayMenu->addAction("Reverse Animate",[this]{runReplay(true);});
                                                                                                                                                                                                                                                replayBtn->setMenu(replayMenu);
                                                                                                                                                                                                                                                stopBtn=new QPushButton("Stop"); stopBtn->setStyleSheet(surfBtnStyle()); stopBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                exportBtn=new QPushButton("Export"); exportBtn->setStyleSheet(mainBtnStyle()); exportBtn->setCursor(Qt::PointingHandCursor);
                                                                                                                                                                                                                                                auto* exportMenu=new QMenu(); exportMenu->setStyleSheet(menuStyle());
                                                                                                                                                                                                                                                exportBtn->setMenu(exportMenu);
                                                                                                                                                                                                                                                ctrlLo->addWidget(startBtn);ctrlLo->addWidget(replayBtn);ctrlLo->addWidget(stopBtn);ctrlLo->addWidget(exportBtn);
                                                                                                                                                                                                                                                stopBtn->setEnabled(false); replayBtn->setEnabled(false);
                                                                                                                                                                                                                                                exportBtn->setEnabled(false); startBtn->setEnabled(false);
                                                                                                                                                                                                                                                mainLo->addLayout(ctrlLo);

                                                                                                                                                                                                                                                statusBar=new QLabel("Ready"); statusBar->setStyleSheet("color:#A0A0A0;padding:6px 12px;font-size:12px;");
                                                                                                                                                                                                                                                mainLo->addWidget(statusBar);
                                                                                                                                                                                                                                                progress=new QProgressBar(); progress->setStyleSheet(progressStyle());
                                                                                                                                                                                                                                                mainLo->addWidget(progress);

                                                                                                                                                                                                                                                connect(basePanel,&MediaPanel::mediaLoaded,this,&ImderGUI::onBaseLoaded);
                                                                                                                                                                                                                                                connect(basePanel,&MediaPanel::mediaCleared,this,&ImderGUI::onBaseCleared);
                                                                                                                                                                                                                                                connect(targetPanel,&MediaPanel::mediaLoaded,this,&ImderGUI::onTargetLoaded);
                                                                                                                                                                                                                                                connect(targetPanel,&MediaPanel::mediaCleared,this,[this]{checkReady();});
                                                                                                                                                                                                                                                connect(startBtn,&QPushButton::clicked,this,[this]{startProcess("preview");});
                                                                                                                                                                                                                                                connect(stopBtn,&QPushButton::clicked,this,&ImderGUI::stopProcess);
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                        public slots:
                                                                                                                                                                                                                                            void onResolutionChanged(const QString& text){
                                                                                                                                                                                                                                                if(text=="Custom"){
                                                                                                                                                                                                                                                    bool ok; int v=QInputDialog::getInt(this,"Custom Resolution","Enter resolution (1-16384):",1024,1,16384,1,&ok);
                                                                                                                                                                                                                                                    if(ok){
                                                                                                                                                                                                                                                        QString nr=QString("%1x%1").arg(v);
                                                                                                                                                                                                                                                        int ci=resCombo->findText("Custom");
                                                                                                                                                                                                                                                        int ei=resCombo->findText(nr);
                                                                                                                                                                                                                                                        if(ei>=0) resCombo->setCurrentIndex(ei);
                                                                                                                                                                                                                                                        else{resCombo->insertItem(ci,nr);resCombo->setCurrentIndex(ci);}
                                                                                                                                                                                                                                                    } else resCombo->setCurrentIndex(0);
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void onModeChanged(const QString& text){
                                                                                                                                                                                                                                                QString mode=text.toLower();
                                                                                                                                                                                                                                                static const QStringList maskModes={"pattern","disguise","navigate","swap","blend","fusion"};
                                                                                                                                                                                                                                                if(mode=="drawer"){
                                                                                                                                                                                                                                                    basePanel->setDrawerMode(true);
                                                                                                                                                                                                                                                    targetPanel->setEnabled(true);
                                                                                                                                                                                                                                                    reverseBtn->setEnabled(false);
                                                                                                                                                                                                                                                    targetPanel->analyzeBtn->setVisible(false);
                                                                                                                                                                                                                                                    targetPanel->penBtn->setVisible(false);
                                                                                                                                                                                                                                                    targetPanel->stopAnalysis();
                                                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                                                    basePanel->setDrawerMode(false);
                                                                                                                                                                                                                                                    targetPanel->setEnabled(!basePanel->filePath.isEmpty());
                                                                                                                                                                                                                                                    targetPanel->analyzeBtn->setVisible(maskModes.contains(mode));
                                                                                                                                                                                                                                                    targetPanel->penBtn->setVisible(maskModes.contains(mode));
                                                                                                                                                                                                                                                    if(!maskModes.contains(mode)) targetPanel->stopAnalysis();
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                checkReady();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void swapMedia(){
                                                                                                                                                                                                                                                if(modeCombo->currentText().toLower()=="drawer") return;
                                                                                                                                                                                                                                                QString bPath=basePanel->filePath; int bRot=basePanel->rotateSteps; bool bFlip=basePanel->isFlipped;
                                                                                                                                                                                                                                                QString tPath=targetPanel->filePath; int tRot=targetPanel->rotateSteps; bool tFlip=targetPanel->isFlipped;
                                                                                                                                                                                                                                                basePanel->setMediaData(tPath,tRot,tFlip);
                                                                                                                                                                                                                                                targetPanel->setMediaData(bPath,bRot,bFlip);
                                                                                                                                                                                                                                                checkReady();
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void onBaseLoaded(const QString&){ targetPanel->setEnabled(true); checkReady(); }
                                                                                                                                                                                                                                            void onBaseCleared(){ targetPanel->setEnabled(false); targetPanel->clearMedia(); checkReady(); }
                                                                                                                                                                                                                                            void onTargetLoaded(const QString&){ checkReady(); }

                                                                                                                                                                                                                                            void checkReady(){
                                                                                                                                                                                                                                                QString mode=modeCombo->currentText().toLower();
                                                                                                                                                                                                                                                bool ready=(mode=="drawer")?!targetPanel->filePath.isEmpty()
                                                                                                                                                                                                                                                :(!basePanel->filePath.isEmpty()&&!targetPanel->filePath.isEmpty());
                                                                                                                                                                                                                                                startBtn->setEnabled(ready); exportBtn->setEnabled(ready);
                                                                                                                                                                                                                                                if(mode!="drawer") reverseBtn->setEnabled(ready);
                                                                                                                                                                                                                                                if(ready){
                                                                                                                                                                                                                                                    auto* m=qobject_cast<QMenu*>(exportBtn->menu());
                                                                                                                                                                                                                                                    m->clear();
                                                                                                                                                                                                                                                    m->addAction("Frame",  [this]{startProcess("export_image");});
                                                                                                                                                                                                                                                    m->addAction("Animation",[this]{startProcess("export_video");});
                                                                                                                                                                                                                                                    m->addAction("GIF",    [this]{startProcess("export_gif");});
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            bool validate(){
                                                                                                                                                                                                                                                QString mode=modeCombo->currentText().toLower();
                                                                                                                                                                                                                                                if(mode=="drawer") return !targetPanel->filePath.isEmpty();
                                                                                                                                                                                                                                                if(basePanel->filePath.isEmpty()||targetPanel->filePath.isEmpty()) return false;
                                                                                                                                                                                                                                                static const QStringList needShape={"pattern","disguise","navigate","swap","blend"};
                                                                                                                                                                                                                                                if(needShape.contains(mode)){
                                                                                                                                                                                                                                                    bool hasMask=!targetPanel->manualMask.empty();
                                                                                                                                                                                                                                                    bool hasAuto=targetPanel->isAnalyzing&&!targetPanel->selectedSegments.empty();
                                                                                                                                                                                                                                                    if(!hasMask&&!hasAuto){
                                                                                                                                                                                                                                                        QMessageBox::warning(this,"Selection Required",
                                                                                                                                                                                                                                                                QString("For %1 mode, please select a shape on Target image (via Analyze or Pen tool).").arg(mode));
                                                                                                                                                                                                                                                        return false;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                return true;
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void startProcess(const QString& mode){
                                                                                                                                                                                                                                                if(!validate()) return;
                                                                                                                                                                                                                                                statusBar->setText("Processing..."); progress->setValue(0);
                                                                                                                                                                                                                                                setProcessingState(true); cachedFrames.clear();
                                                                                                                                                                                                                                                currentFps=fpsCombo->currentText().toInt();
                                                                                                                                                                                                                                                ProcessConfig cfg;
                                                                                                                                                                                                                                                cfg.basePath=basePanel->filePath.toStdString();
                                                                                                                                                                                                                                                cfg.tgtPath=targetPanel->filePath.toStdString();
                                                                                                                                                                                                                                                cfg.mode=mode.toStdString();
                                                                                                                                                                                                                                                cfg.algo=modeCombo->currentText().toLower().toStdString();
                                                                                                                                                                                                                                                cfg.rotBase=basePanel->rotateSteps; cfg.flipBase=basePanel->isFlipped;
                                                                                                                                                                                                                                                cfg.rotTgt=targetPanel->rotateSteps; cfg.flipTgt=targetPanel->isFlipped;
                                                                                                                                                                                                                                                cfg.fps=currentFps;
                                                                                                                                                                                                                                                cfg.resolution=resCombo->currentText().split('x')[0].toInt();
                                                                                                                                                                                                                                                cfg.soundOpt="mute"; cfg.audioQuality=30;
                                                                                                                                                                                                                                                if(cfg.algo=="drawer"){
                                                                                                                                                                                                                                                    cfg.baseImageArray=basePanel->getDrawingArray();
                                                                                                                                                                                                                                                    if(cfg.baseImageArray.empty()){statusBar->setText("No drawing found");setProcessingState(false);return;}
                                                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                                                    if(!targetPanel->manualMask.empty()) cfg.mask=targetPanel->getMask();
                                                                                                                                                                                                                                                    else cfg.mask=targetPanel->getMask();
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                worker=new ProcessingThread(cfg,this);
                                                                                                                                                                                                                                                connect(worker,&ProcessingThread::progressSignal,progress,&QProgressBar::setValue);
                                                                                                                                                                                                                                                connect(worker,&ProcessingThread::frameSignal,this,&ImderGUI::updatePreviewFrame);
                                                                                                                                                                                                                                                connect(worker,&ProcessingThread::finishedSignal,this,&ImderGUI::processFinished);
                                                                                                                                                                                                                                                connect(worker,&ProcessingThread::errorSignal,this,&ImderGUI::processError);
                                                                                                                                                                                                                                                connect(worker,&QThread::finished,worker,&QObject::deleteLater);
                                                                                                                                                                                                                                                worker->start();
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            void stopProcess(){ if(worker){worker->stop();statusBar->setText("Stopping...");} }

                                                                                                                                                                                                                                            void updatePreviewFrame(const QImage& qi){
                                                                                                                                                                                                                                                previewDisplay->setPixmap(QPixmap::fromImage(qi));
                                                                                                                                                                                                                                                cachedFrames.push_back(qi.copy());
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void processFinished(const QString& msg){
                                                                                                                                                                                                                                                statusBar->setText(msg); setProcessingState(false);
                                                                                                                                                                                                                                                progress->setValue(100); replayBtn->setEnabled(true);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void processError(const QString& err){
                                                                                                                                                                                                                                                statusBar->setText("Error: "+err);
                                                                                                                                                                                                                                                QMessageBox::critical(this,"Error",err); setProcessingState(false);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void setProcessingState(bool proc){
                                                                                                                                                                                                                                                QString mode=modeCombo->currentText().toLower();
                                                                                                                                                                                                                                                startBtn->setEnabled(!proc); exportBtn->setEnabled(!proc);
                                                                                                                                                                                                                                                basePanel->setEnabled(!proc); targetPanel->setEnabled(!proc);
                                                                                                                                                                                                                                                stopBtn->setEnabled(proc);
                                                                                                                                                                                                                                                if(mode!="drawer") reverseBtn->setEnabled(!proc);
                                                                                                                                                                                                                                                modeCombo->setEnabled(!proc); resCombo->setEnabled(!proc);
                                                                                                                                                                                                                                                replayBtn->setEnabled(false);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void runReplay(bool reverse){
                                                                                                                                                                                                                                                if(cachedFrames.empty()) return;
                                                                                                                                                                                                                                                replayFrames=reverse?std::vector<QImage>(cachedFrames.rbegin(),cachedFrames.rend()):cachedFrames;
                                                                                                                                                                                                                                                replayStartMs=QDateTime::currentMSecsSinceEpoch();
                                                                                                                                                                                                                                                replayDuration=10.0;
                                                                                                                                                                                                                                                if(replayTimer){replayTimer->stop();delete replayTimer;}
                                                                                                                                                                                                                                                replayTimer=new QTimer(this);
                                                                                                                                                                                                                                                connect(replayTimer,&QTimer::timeout,this,&ImderGUI::playFrameSynced);
                                                                                                                                                                                                                                                replayTimer->start(16);
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                            void playFrameSynced(){
                                                                                                                                                                                                                                                double elapsed=(QDateTime::currentMSecsSinceEpoch()-replayStartMs)/1000.0;
                                                                                                                                                                                                                                                double prog=elapsed/replayDuration;
                                                                                                                                                                                                                                                int total=(int)replayFrames.size();
                                                                                                                                                                                                                                                int idx=(int)(prog*total);
                                                                                                                                                                                                                                                if(idx<total) previewDisplay->setPixmap(QPixmap::fromImage(replayFrames[idx]));
                                                                                                                                                                                                                                                else{ previewDisplay->setPixmap(QPixmap::fromImage(replayFrames.back())); replayTimer->stop(); }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                        int main(int argc,char* argv[]){
                                                                                                                                                                                                                                            QApplication app(argc,argv);
                                                                                                                                                                                                                                            app.setStyle("Fusion");

                                                                                                                                                                                                                                            bool isCLI=false;
                                                                                                                                                                                                                                            int argOffset=1;

                                                                                                                                                                                                                                            if(argc>1){
                                                                                                                                                                                                                                                if(std::string(argv[1])=="cli"){
                                                                                                                                                                                                                                                    if(argc==2){ interactiveCLI(); return 0; }
                                                                                                                                                                                                                                                    else{ isCLI=true; argOffset=2; }
                                                                                                                                                                                                                                                } else if(argc>=3){ isCLI=true; argOffset=1; }
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            if(isCLI){
                                                                                                                                                                                                                                                if(argc-argOffset<2){
                                                                                                                                                                                                                                                    fprintf(stderr,"Error: Missing required arguments. Usage:\n");
                                                                                                                                                                                                                                                    fprintf(stderr,"  imder <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]\n");
                                                                                                                                                                                                                                                    fprintf(stderr,"  imder cli [interactive mode]\n");
                                                                                                                                                                                                                                                    fprintf(stderr,"\nSound options: mute, sound, target-sound\n");
                                                                                                                                                                                                                                                    fprintf(stderr,"Quality: 1-10 (only for target-sound, default 3)\n");
                                                                                                                                                                                                                                                    return 1;
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                std::string basePath=argv[argOffset];
                                                                                                                                                                                                                                                std::string tgtPath =argv[argOffset+1];
                                                                                                                                                                                                                                                bool bIsVid=isVideoFile(basePath), tIsVid=isVideoFile(tgtPath);

                                                                                                                                                                                                                                                std::string algo="merge";
                                                                                                                                                                                                                                                if(argc-argOffset>=3){
                                                                                                                                                                                                                                                    algo=argv[argOffset+2];
                                                                                                                                                                                                                                                    std::transform(algo.begin(),algo.end(),algo.begin(),::tolower);
                                                                                                                                                                                                                                                    if(bIsVid||tIsVid){
                                                                                                                                                                                                                                                        if(algo!="merge"&&algo!="shuffle"&&algo!="missform"){
                                                                                                                                                                                                                                                            algo="merge";
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                    } else {
                                                                                                                                                                                                                                                        static const std::vector<std::string> valid={"shuffle","merge","missform","fusion"};
                                                                                                                                                                                                                                                        if(std::find(valid.begin(),valid.end(),algo)==valid.end()){
                                                                                                                                                                                                                                                            fprintf(stderr,"Error: Invalid algorithm '%s'. Valid options: shuffle, merge, missform, fusion\n",algo.c_str());
                                                                                                                                                                                                                                                            return 1;
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                int resolution=512;
                                                                                                                                                                                                                                                if(argc-argOffset>=4){
                                                                                                                                                                                                                                                    try{ resolution=std::stoi(argv[argOffset+3]); }
                                                                                                                                                                                                                                                    catch(...){ fprintf(stderr,"Error: Resolution must be a number\n"); return 1; }
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                std::string soundOpt="mute"; int audioQuality=30;
                                                                                                                                                                                                                                                if(argc-argOffset>=5){
                                                                                                                                                                                                                                                    soundOpt=argv[argOffset+4];
                                                                                                                                                                                                                                                    std::transform(soundOpt.begin(),soundOpt.end(),soundOpt.begin(),::tolower);
                                                                                                                                                                                                                                                    if(soundOpt=="target-sound"&&!tIsVid){
                                                                                                                                                                                                                                                        fprintf(stderr,"Error: Target Sound option requires target to be a video file.\n"); return 1;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    if(soundOpt!="mute"&&soundOpt!="sound"&&soundOpt!="target-sound"){
                                                                                                                                                                                                                                                        soundOpt="mute";
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    if(argc-argOffset>=6&&soundOpt=="target-sound"){
                                                                                                                                                                                                                                                        try{
                                                                                                                                                                                                                                                            int q=std::stoi(argv[argOffset+5]);
                                                                                                                                                                                                                                                            if(q>=1&&q<=10) audioQuality=q*10;
                                                                                                                                                                                                                                                            else{ audioQuality=30; }
                                                                                                                                                                                                                                                        } catch(...){}
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                cliProcessAndExport(basePath,tgtPath,algo,resolution,soundOpt,audioQuality);
                                                                                                                                                                                                                                                return 0;
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                            ImderGUI win;
                                                                                                                                                                                                                                            win.show();
                                                                                                                                                                                                                                            return app.exec();
                                                                                                                                                                                                                                        }

                                                                                                                                                                                                                                        #include "imder_linux.moc"
