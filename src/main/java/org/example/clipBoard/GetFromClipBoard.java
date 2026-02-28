    package org.example.clipBoard;

    import org.example.link.NativeLib;


    import java.awt.*;
    import java.awt.datatransfer.Clipboard;
    import java.awt.datatransfer.DataFlavor;
    import java.awt.datatransfer.Transferable;
    import java.awt.image.BufferedImage;

    import java.awt.image.DataBufferByte;

    public class GetFromClipBoard {
        static final String modelPath = "C:\\Users\\zizem\\OneDrive\\Desktop\\cource\\models\\LapSRN_x4.pb";
        static final String outputPath = "C:\\Users\\zizem\\OneDrive\\Desktop\\cource\\try.png"; //DEPRICATED

        public static void Get() throws Exception {
            try {
                //Отримуємо данні з буфера обміну
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                Transferable contents = clipboard.getContents(null);

                if (contents != null && contents.isDataFlavorSupported(DataFlavor.imageFlavor)) {
                    //Image - сирі данні для опенСв і через це вказуємо тип каналів(B G R)
                    Image image = (Image) contents.getTransferData(DataFlavor.imageFlavor);
                    BufferedImage bufferedImage = new BufferedImage(
                            image.getWidth(null),
                            image.getHeight(null),
                            BufferedImage.TYPE_3BYTE_BGR);

                    Graphics2D g2d = bufferedImage.createGraphics();
                    g2d.drawImage(image, 0, 0, null);
                    g2d.dispose();

                    //Сирі пікселі з памяті як масив байтів
                    byte[] imageBytes = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();

                    //Визов С++ модуля
                    NativeLib.INSTANCE.processImage(imageBytes, bufferedImage.getWidth(), bufferedImage.getHeight(), 3, modelPath, outputPath);

                } else{

                    System.out.println("Clipboard does not contain an image");
                }

            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                e.printStackTrace();
            }

        }
    }

